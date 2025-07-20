from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API")
client = genai.Client(api_key=GOOGLE_API_KEY)

# Define activity prompts
ACTIVITY_PROMPTS = {
    "casual": "Start a friendly conversation with the driver to keep them engaged and awake.",
    "riddle": "Ask the driver a fun riddle and wait for their answer.",
    "story": "Tell the driver a short, interesting story to stimulate their mind.",
    "brain_teaser": "Pose a quick brain teaser or puzzle for the driver.",
    "number_plate": "Challenge the driver to spot and remember the numbers on the car in front, or calculate sums or products with the number digits on the licence plate."
}

def get_activity_prompt(activity_type):
    return ACTIVITY_PROMPTS.get(activity_type, ACTIVITY_PROMPTS["casual"])

def run_activity(activity_type):
    prompt = get_activity_prompt(activity_type)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    print(f"\n--- {activity_type.upper()} ---")
    print(response.text)

# Example usage:
if __name__ == "__main__":
    # Simulate picking an activity based on fatigue level or preference
    for activity in ["casual", "riddle", "story", "brain_teaser", "number_plate"]:
        run_activity(activity)