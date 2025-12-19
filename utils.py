def generate_communication_template(incident_number, short_desc, state, assignment_group, impact_details):
    """
    Generates a formatted Major Incident communication string.
    """
    template = f"""Major Incident Update: 

{incident_number}

 - 

{short_desc}


Current Status: 

{state}


Team Owning: 

{assignment_group}


Impact: 

{impact_details}
"""
    return template
