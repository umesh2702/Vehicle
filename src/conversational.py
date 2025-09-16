from typing import List, Optional

def format_dtc_response(dtc_info: dict) -> str:
    if not dtc_info:
        return "I couldn't find details for the reported code."
    resp = f"Detected trouble code {dtc_info['code']}. {dtc_info.get('meaning','')}"
    if dtc_info.get('possible_cause'):
        resp += f". Possible cause: {dtc_info['possible_cause']}"
    if dtc_info.get('fix_suggestion'):
        resp += f". Suggested fix: {dtc_info['fix_suggestion']}"
    if dtc_info.get('urgency'):
        resp += f". Urgency: {dtc_info.get('urgency')}"
    return resp

def format_anomaly_response(anomalies: List[str]) -> str:
    if not anomalies:
        return "No immediate anomalies detected in the provided snapshot."
    out = "I detected the following issues:"
    for a in anomalies:
        out += f"\n- {a}"
    return out

def assemble_reply(dtc_info: Optional[dict], anomalies: List[str], recommended_actions: List[str] = None) -> str:
    parts = []
    if dtc_info:
        parts.append(format_dtc_response(dtc_info))
    parts.append(format_anomaly_response(anomalies))
    if recommended_actions:
        parts.append("Recommended next steps: " + " ".join(recommended_actions))
    parts.append("If you'd like, I can simplify this explanation or suggest checks to do before visiting a mechanic.")
    return "\n\n".join(parts)
