from google import genai
import os
from dotenv import load_dotenv
from enum import Enum

class FatigueLevel(Enum):
    NONE = "None"
    LOW = "Low"
    MEDIUM_LOW = "Medium Low"
    MEDIUM = "Medium"
    MEDIUM_HIGH = "Medium High"
    HIGH = "High"
    EXTREME = "Extreme"

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API")
client = genai.Client(api_key=GOOGLE_API_KEY)
STARTING_PROMPT = """
You are a helpful assistant designed to keep a driver engaged and alert
during long drives. Based on the driver's fatigue level, suggest an
activity that is appropriate for their current state. The activities
should be engaging, stimulating, and help maintain focus on the road."""

# Define activity prompts
ACTIVITY_PROMPTS = {
    "casual": "Start a friendly conversation with the driver to keep them engaged and awake.",
    "riddle": "Ask the driver a fun riddle and wait for their answer.",
    "story": "Tell the driver a short, interesting story to stimulate their mind.",
    "brain_teaser": "Pose a quick brain teaser or puzzle for the driver.",
    "word_game": "Engage the driver in a word game, like 'I Spy' or '20 Questions'.",
    "math_game": "Challenge the driver with a simple math problem or a mental calculation.",
    "number_plate": "Challenge the driver to spot and remember the numbers on the car in front, or calculate sums or products with the number digits on the licence plate."
}

def get_activity_prompt(activity_type):
    return ACTIVITY_PROMPTS.get(activity_type, ACTIVITY_PROMPTS["casual"])

def get_prompt(activity_type, fatigue_level=None):
    if fatigue_level is None:
        fatigue_level = FatigueLevel.NONE
    elif fatigue_level == "low":
        fatigue_level = FatigueLevel.LOW
    elif fatigue_level == "medium low":
        fatigue_level = FatigueLevel.MEDIUM_LOW
    elif fatigue_level == "medium":
        fatigue_level = FatigueLevel.MEDIUM
    elif fatigue_level == "medium high":
        fatigue_level = FatigueLevel.MEDIUM_HIGH
    elif fatigue_level == "high":
        fatigue_level = FatigueLevel.HIGH
    else:
        fatigue_level = FatigueLevel.NONE
    return f"{STARTING_PROMPT} The driver is currently at a {fatigue_level} fatigue level. {get_activity_prompt(activity_type)}"

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