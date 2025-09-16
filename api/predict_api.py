import os
import re
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import cohere
from src.models import predict_vehicle_health
from src.dtc_lookup import DTCLookup

# Load environment variables
load_dotenv()
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
co = cohere.Client(COHERE_API_KEY)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize DTC lookup
dtc_lookup = DTCLookup()

def ask_cohere(prompt: str) -> str:
    """Generate a polished DTC explanation using Cohere's Chat API."""
    try:
        response = co.chat(
            model="command-a-03-2025",   # <-- new active model
            message=prompt,
            temperature=0.7,
            max_tokens=200
        )
        return response.text.strip()
    except Exception as e:
        return f"(⚠️ Cohere API error: {e})"


def parse_message(user_message: str):
    """Extract RPM, speed, temp, and DTC code from user message."""
    user_message = user_message.upper()

    dtc_match = re.search(r'(P\d{4})', user_message)
    dtc_code = dtc_match.group(1) if dtc_match else None

    rpm_match = re.search(r'(\d+)\s*RPM', user_message)
    rpm = int(rpm_match.group(1)) if rpm_match else None

    temp_match = re.search(r'(\d+)\s*°?C', user_message)
    temp = int(temp_match.group(1)) if temp_match else None

    speed_match = re.search(r'(\d+)\s*(KMH|KMPH|KM/H|SPEED)', user_message)
    speed = int(speed_match.group(1)) if speed_match else None

    return rpm, speed, temp, dtc_code

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    user_message = data.get("message", "")

    rpm, speed, temp, dtc_code = parse_message(user_message)

    # Predict vehicle health
    if dtc_code:
        dtc_info = dtc_lookup.lookup(dtc_code)
        dtc_dict = {dtc_code: dtc_info['meaning']} if dtc_info else {}
        health_prediction = predict_vehicle_health(rpm, speed, temp, dtc_code, dtc_dict)
    else:
        dtc_dict = {}
        health_prediction = predict_vehicle_health(rpm, speed, temp, None, {})

    # Generate chatbot response
    if dtc_code and dtc_info:
        # Known DTC code → format neatly with Cohere
        prompt = (
            f"You are a professional car assistant. Explain the DTC code in a **friendly, easy-to-read way**.\n"
            f"Include the following details neatly in separate lines: Meaning, Possible Causes, Fix Suggestion, Urgency.\n"
            f"Use emojis and bullets to highlight key points. Keep it concise and cool (6-8 lines).\n\n"
            f"DTC Code: {dtc_info['code']}\n"
            f"Meaning: {dtc_info['meaning']}\n"
            f"Possible Causes: {dtc_info['possible_cause']}\n"
            f"Fix Suggestion: {dtc_info['fix_suggestion']}\n"
            f"Urgency: {dtc_info['urgency']}"
        )
        result_message = ask_cohere(prompt)

    else:
        # Unknown DTC or no DTC → Cohere generates a friendly explanation from user message
        prompt = (
            f"You are a professional car assistant. The user reported the following info:\n"
            f"\"{user_message}\"\n\n"
            f"Provide a **concise, friendly, helpful response** (6-8 lines), highlighting key points "
            f"with emojis or bullets. Include meaning, possible causes, fix suggestions, and urgency if possible."
        )
        result_message = ask_cohere(prompt)

    return jsonify({
        "message": result_message,
        "health_prediction": health_prediction
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)