def generate_communication_template(incident_number, short_desc, state, assignment_group, impact_details,
                                    priority=None, opened_at=None, description_snippet=None,
                                    cluster_info=None, suspect_changes=None):
    """
    Generates a formatted Major Incident communication template.
    Enriched with optional: priority, opened_at, cluster context, suspect changes.
    """
    import pandas as pd

    def _fmt_ts(ts):
        if ts is None:
            return ""
        try:
            if hasattr(pd, 'isna') and pd.isna(ts):
                return ""
            t = pd.to_datetime(ts)
            return t.strftime("%Y-%m-%d %H:%M")
        except Exception:
            return str(ts) if ts else ""

    # Build sections
    lines = []
    lines.append("=" * 50)
    lines.append("MAJOR INCIDENT UPDATE")
    lines.append("=" * 50)
    lines.append("")
    lines.append(f"Incident: {incident_number}")
    lines.append(f"Summary: {short_desc or '(No description)'}")
    lines.append("")

    # Metadata block
    lines.append("--- Current Status ---")
    lines.append(f"Status: {state or 'Unknown'}")
    if priority:
        lines.append(f"Priority: {priority}")
    if opened_at:
        lines.append(f"Opened: {_fmt_ts(opened_at)}")
    lines.append(f"Team Owning: {assignment_group or 'Unassigned'}")
    lines.append("")

    # Impact (user-provided)
    lines.append("--- Impact ---")
    lines.append(impact_details if impact_details else "[Add impact details — users affected, systems down, business impact]")
    lines.append("")

    # Cluster context (related incidents)
    if cluster_info:
        cid = cluster_info.get('cluster_id')
        size = cluster_info.get('size', 0)
        related = cluster_info.get('related_incidents', [])
        if size > 1:
            lines.append("--- Related Incidents (same cluster) ---")
            lines.append(f"Cluster size: {size} incidents")
            if related:
                # Show up to 10, or summarize
                if len(related) <= 10:
                    lines.append("Incidents: " + ", ".join(related))
                else:
                    lines.append("Incidents: " + ", ".join(related[:5]) + f" ... and {len(related) - 5} more")
            lines.append("")

    # Suspect changes (potential root cause)
    if suspect_changes:
        lines.append("--- Potential Root Cause (recent changes) ---")
        for chg in suspect_changes[:5]:  # Top 5
            lines.append(f"  • {chg}")
        lines.append("")

    # Description snippet if meaningful
    if description_snippet and len(str(description_snippet).strip()) > 20:
        snippet = str(description_snippet).strip()[:300]
        if len(str(description_snippet)) > 300:
            snippet += "..."
        lines.append("--- Description ---")
        lines.append(snippet)
        lines.append("")

    # Action placeholders
    lines.append("--- Actions Taken ---")
    lines.append("[Add: steps taken, workarounds implemented]")
    lines.append("")
    lines.append("--- Next Steps ---")
    lines.append("[Add: planned actions, ETA for resolution]")
    lines.append("")
    lines.append("--- ETA ---")
    lines.append("[Add: expected resolution time]")
    lines.append("")
    lines.append("=" * 50)

    return "\n".join(lines)
