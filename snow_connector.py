import pandas as pd
import datetime

def get_snow_data(table_name):
    """
    Mock function to simulate fetching data from ServiceNow.
    Returns a list of dictionaries (JSON-like) based on the table name.
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
                "assignment_group": "Email Team"
            },
            {
                "number": "INC001002",
                "short_description": "Slow VPN connection",
                "description": "VPN latency is high.",
                "opened_at": yesterday.isoformat(),
                "closed_at": "",
                "state": "In Progress",
                "assignment_group": "Network Team"
            },
            {
                "number": "INC001003",
                "short_description": "Printer jam on 2nd floor",
                "description": "Paper jam.",
                "opened_at": now.isoformat(),
                "closed_at": "",
                "state": "New",
                "assignment_group": "Hardware Team"
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
