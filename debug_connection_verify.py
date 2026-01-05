import os
import snow_connector
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get credentials
instance_url = os.getenv("SNOW_INSTANCE_URL")
username = os.getenv("SNOW_USERNAME")
password = os.getenv("SNOW_PASSWORD")

if not instance_url or not username or not password:
    print("Error: Missing ServiceNow credentials in .env")
    exit(1)

print(f"Connecting to {instance_url} as {username}...")

try:
    client = snow_connector.ServiceNowClient(instance_url, username, password)
    
    # Try fetching one incident
    print("Fetching incidents...")
    incidents = client.fetch_table_data('incident', limit=1)
    
    if incidents:
        print(f"Success! Fetched {len(incidents)} incident(s).")
        print("Sample:", incidents[0]['number'])
    else:
        print("Success! Connected but no incidents found.")

except Exception as e:
    print(f"Connection Failed: {e}")
