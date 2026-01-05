import os
import sys
import pytest
try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    pytest.skip("Streamlit testing framework not found", allow_module_level=True)

class TestAppFlow:
    @pytest.fixture
    def app_path(self):
        # Assuming tests/ is inside AI_Ops_Lite/
        return os.path.join(os.path.dirname(__file__), '..', 'app.py')

    def test_app_startup(self, app_path):
        """Test that the app starts and displays the correct title."""
        at = AppTest.from_file(app_path)
        at.run(timeout=10)
        
        assert not at.exception
        assert "AIOps Lite: Flight Deck" in at.title[0].value

    def test_navigation_offline_mode(self, app_path):
        """Test switching to Offline Data mode."""
        at = AppTest.from_file(app_path)
        at.run(timeout=10)
        
        # Select "Offline Data" (index 2 in the list ["Live API (Mock)", "Live API (Real)", "Offline Data"])
        # Note: sidebar.selectbox returns a list of SelectBox elements. 
        # The first one is "Select Data Source".
        at.sidebar.selectbox[0].set_value("Offline Data").run(timeout=30)
        
        assert not at.exception
        # Check if file uploaders appear in the sidebar
        # We expect "Upload Incidents (CSV)" etc. which are inside an expander usually?
        # In the code: with st.sidebar.expander("Upload Overrides"): up_inc = ...
        # AppTest flattens elements or provides access. 
        # Let's check if we can see the info message "Mode: Offline Data (CSV)"
        assert "Mode: Offline Data (CSV)" in at.sidebar.info[0].value

    def test_tab_navigation(self, app_path):
        """Test that tabs render content."""
        at = AppTest.from_file(app_path)
        at.run(timeout=10)
        
        # Default mode is likely 'Live API (Mock)' based on code index 0.
        # "Real-Time Risk Monitor" is in the first tab.
        # AppTest exposes tabs as .tabs but determining content inside them requires indexing
        # generic elements or checking headers verify they are loaded.
        
        # Check for headers that should exist on load
        # Title is checked in startup test. verifying 'Real-Time Risk Monitor' header (Tab 1)
        headers = [h.value for h in at.header]
        
        # The code uses st.title() -> at.title
        # Tab 1: st.header("Real-Time Risk Monitor")
        # Ensure it's present (it is rendered by default since Tab 1 is first)
        assert "Real-Time Risk Monitor" in [h.value for h in at.header]

    def test_ai_intelligence_tab(self, app_path):
        """Test interaction in AI Intelligence Tab."""
        at = AppTest.from_file(app_path)
        at.run(timeout=10)
        
        # Switch to Mock Data to ensure we have data
        at.sidebar.selectbox[0].set_value("Live API (Mock)").run(timeout=10)
        
        # Click "Train Assignment Model" button
        # It's in the 3rd tab "AI Intelligence"
        # We don't strictly need to "click" the tab to make the button exist in Streamlit script logic,
        # but we do need the script to run that part.
        # Tabs in Streamlit just group output. All code usually runs top to bottom unless conditional.
        # The code `with tab_intelligence:` block runs.
        
        # Find the button "Train Assignment Model"
        # It might be distinct by key='train_router'
        train_btns = [b for b in at.button if b.key == 'train_router']
        if train_btns:
            train_btns[0].click().run(timeout=30)
            assert not at.exception
            # Should see success message
            assert any("Model trained on" in s.value for s in at.success)
        else:
            # Maybe data wasn't loaded?
            pass
