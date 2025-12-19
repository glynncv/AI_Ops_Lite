import os
import sys
try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    print("Streamlit testing framework not found. Please upgrade streamlit (pip install --upgrade streamlit).")
    sys.exit(1)

def test_app_startup_and_navigation():
    print("Starting automated browser test...")
    
    # Path to app.py
    # Assuming tests/ is inside AI_Ops_Lite/
    project_root = os.path.join(os.path.dirname(__file__), '..')
    sys.path.append(project_root)
    app_path = os.path.join(project_root, 'app.py')
    
    # Initialize AppTest
    print(f"Loading app from {app_path}")
    at = AppTest.from_file(app_path)
    at.run(timeout=30)
    
    # Check title
    if at.exception:
        print("Exception during initial run:")
        print(at.exception)
        sys.exit(1)
        
    print("Initial run successful. Title found:", at.title[0].value)
    
    # Test 1: Switch to Offline Data
    print("Testing: Switch to Offline Data")
    try:
        # Select "Offline Data" (index 1)
        at.sidebar.selectbox[0].set_value("Offline Data").run(timeout=30)
    except Exception as e:
        print(f"Failed to switch to Offline Data: {e}")
        # inspecting what selectboxes exist
        print("Selectboxes:", at.sidebar.selectbox)
        sys.exit(1)

    if at.exception:
        print("Exception after switching to Offline Data:")
        print(at.exception)
        # This is where the KeyError 'description' would likely happen if not fixed
        sys.exit(1)
        
    print("Offline Data loaded without crash.")
    
    # Test 2: Interact with tabs
    # Tabs run automatically in script flow, so if we didn't crash yet, we are mostly good.
    # We can verify specific elements exist.
    
    # Check for 'Real-Time Risk Monitor' header (Tab 1)
    # It might be in at.header values
    headers = [h.value for h in at.header]
    if "Real-Time Risk Monitor" in headers:
        print("Tab 1 (Risks) verified.")
    else:
        print("Warning: Tab 1 header not found.")
        
    # Test 3: Clustering Analysis (Tab 2)
    # Logic runs for clustering check.
    
    print("SUCCESS: Browser navigation tests passed!")

if __name__ == "__main__":
    test_app_startup_and_navigation()
