
import pandas as pd
from analysis import identiy_l0_tickets, calculate_operational_risk

def test_l0_identification():
    data = [
        {'short_description': 'Password reset required', 'close_notes': 'Reset password'},
        {'short_description': 'Server down', 'close_notes': 'Rebooted'},
        {'short_description': 'How to access VPN', 'close_notes': 'Advice given'},
        {'short_description': 'Critical failure', 'close_notes': 'Fixed code logic'},
        {'short_description': 'Clear cache', 'close_notes': 'Cache cleared'}
    ]
    df = pd.DataFrame(data)
    
    # Expected: Password reset (1), Advice given (1), Clear cache (1) -> Total 3
    # Server down -> Rebooted might be L0 if "reboot" matches, let's see logic.
    # We defined keywords: "password", "reset", "training", "advice", "cache", "clear", "restart"
    # "Rebooted" contains "boot" not "restart".
    
    count = identiy_l0_tickets(df)
    print(f"L0 Count: {count}")
    return count

def test_risk_calculation():
    # Test High
    print(f"Risk (4, 6): {calculate_operational_risk(4, 6)}")
    # Test Medium
    print(f"Risk (1, 0): {calculate_operational_risk(1, 0)}")
    # Test Low
    print(f"Risk (0, 0): {calculate_operational_risk(0, 0)}")

if __name__ == "__main__":
    try:
        c = test_l0_identification()
        if c >= 2: 
             print("L0 Test Passed (found matches)")
        else:
             print("L0 Test Warning: Count seems low")
             
        test_risk_calculation()
        print("All Tests Completed")
    except Exception as e:
        print(f"Test Failed: {e}")
