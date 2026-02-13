import pandas as pd
import datetime
import requests
from requests.auth import HTTPBasicAuth

# --- Mock Data Generation (Preserved) ---
def get_mock_data(table_name):
    """
    Returns a list of dictionaries (JSON-like) based on the table name for testing.
    """
    now = datetime.datetime.now()
    yesterday = now - datetime.timedelta(days=1)
    two_days_ago = now - datetime.timedelta(days=2)
    
    if table_name == 'incident':
        return [
            {
                "number": "INC001001",
                "short_description": "Email server down",
                "description": "Users cannot send emails.",
                "opened_at": two_days_ago.isoformat(),
                "closed_at": yesterday.isoformat(),
                "state": "Closed",
                "assignment_group": "Email Team",
                "priority": "2 - High",
                "location": ""
            },
            {
                "number": "INC001002",
                "short_description": "Slow VPN connection",
                "description": "VPN latency is high.",
                "opened_at": yesterday.isoformat(),
                "closed_at": "",
                "state": "In Progress",
                "assignment_group": "Network Team",
                "priority": "3 - Moderate",
                "location": ""
            },
            {
                "number": "INC001003",
                "short_description": "Printer jam on 2nd floor",
                "description": "Paper jam.",
                "opened_at": now.isoformat(),
                "closed_at": "",
                "state": "New",
                "assignment_group": "Hardware Team",
                "priority": "4 - Low",
                "location": ""
            }
        ]
    elif table_name == 'problem':
        return [
            {
                "number": "PRB002001",
                "short_description": "Recurring Email Server Outages",
                "description": "Root cause analysis for email server.",
                "opened_at": two_days_ago.isoformat(),
                "closed_at": "",
                "state": "Open",
                "assignment_group": "Email Team"
            }
        ]
    elif table_name == 'change_request':
        return [
             {
                "number": "CHG003001",
                "short_description": "Patch Email Server",
                "description": "Applying monthly security patches.",
                "opened_at": two_days_ago.isoformat(),
                "closed_at": yesterday.isoformat(),
                "state": "Closed",
                "assignment_group": "Email Team"
            }
        ]
    else:
        return []

# --- Real ServiceNow Client ---
class ServiceNowClient:
    def __init__(self, instance_url, username, password, use_proxy=False, verify_ssl=True):
        self.base_url = instance_url.rstrip('/')
        self.auth = HTTPBasicAuth(username, password)
        self.headers = {"Content-Type": "application/json", "Accept": "application/json"}
        
        # Proxy configuration - set to empty dict to bypass system proxies
        if not use_proxy:
            self.proxies = {}  # Bypass all proxies
        else:
            self.proxies = None  # Use system proxy settings
        
        # SSL verification - can be disabled for corporate environments with self-signed certs
        self.verify_ssl = verify_ssl

    def fetch_table_data(self, table_name, limit=100, query=None, fields=None):
        """
        Generic function to fetch data from a ServiceNow table.
        """
        url = f"{self.base_url}/api/now/table/{table_name}"
        params = {
            "sysparm_limit": limit,
            "sysparm_display_value": "true", # Get readable values
            "sysparm_exclude_reference_link": "true" 
        }
        if query:
            params['sysparm_query'] = query
        if fields:
            params['sysparm_fields'] = fields

        try:
            response = requests.get(
                url, 
                auth=self.auth, 
                headers=self.headers, 
                params=params,
                proxies=self.proxies,  # Use configured proxy settings
                verify=self.verify_ssl,  # SSL verification (can be disabled for corporate certs)
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            result = data.get('result', [])
            return result
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from {table_name}: {e}")
            raise e

    def get_incidents(self, days_back=30, limit=500):
        # Query: Created in last X days
        # Using sys_created_on >= javascript:gs.daysAgo(30) is standard SNOW query syntax
        query = f"opened_at>=javascript:gs.daysAgo({days_back})^ORDERBYDESCopened_at"
        
        # Explicitly request critical fields including priority
        fields = "number,short_description,description,state,priority,opened_at,closed_at,assignment_group,close_notes,location,sys_created_on,sys_updated_on"
        
        return self.fetch_table_data('incident', limit=limit, query=query, fields=fields)

    def get_problems(self, days_back=90, limit=500):
        query = f"opened_at>=javascript:gs.daysAgo({days_back})^ORDERBYDESCopened_at"
        
        # Explicitly request critical fields including priority and location
        fields = "number,short_description,description,state,priority,opened_at,closed_at,assignment_group,close_notes,location,sys_created_on,sys_updated_on"
        
        return self.fetch_table_data('problem', limit=limit, query=query, fields=fields)

    def get_changes(self, days_back=30, limit=500):
        query = f"closed_at>=javascript:gs.daysAgo({days_back})^ORDERBYDESCclosed_at"
        
        # Explicitly request critical fields
        fields = "number,short_description,description,state,closed_at,assignment_group,sys_created_on,sys_updated_on"
        
        return self.fetch_table_data('change_request', limit=limit, query=query, fields=fields)

# --- Unified Accessor ---
def get_snow_data(table_name, client=None):
    """
    Wrapper to get data. If client is provided, use it (Real).
    If client is None, use mock data.
    """
    if client:
        if table_name == 'incident':
            return client.get_incidents()
        elif table_name == 'problem':
            return client.get_problems()
        elif table_name == 'change_request':
            return client.get_changes()
        else:
            return client.fetch_table_data(table_name)
    else:
        return get_mock_data(table_name)
