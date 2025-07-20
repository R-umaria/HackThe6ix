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

load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API")
client = genai.Client(api_key=GOOGLE_API_KEY)
STARTING_PROMPT = """
You are Alex, a helpful assistant designed to keep a driver engaged and
alert during long drives. Based on the driver's fatigue level, suggest
an activity that is appropriate for their current state. The activities
should be engaging, stimulating, and help maintain focus on the road.
YOU will interact with the driver in a friendly and supportive manner.
The driver may ask you questions, and you should respond in a way that 
keeps them engaged and alert. If the driver is feeling ok and asks you
to stop for a while, you should politely end the conversation until they
say 'Hey Alex, let's continue.'. You should also ask for the driver's name
and destination at the start of the conversation to personalize the interaction. 
However, don't be invasive. Keep it cool and casual like a friend would.
Make sure to adapt your responses based on the driver's fatigue level that
it constantly being updated by the car's system.
The fatigue or  drowsiness levels are: None, Low, Medium Low, Medium,
Medium High, and High
"""

# Map fatigue levels to activities
FATIGUE_ACTIVITY_MAP = {
    FatigueLevel.NONE: "casual",
    FatigueLevel.LOW: "word_game",
    FatigueLevel.MEDIUM_LOW: "math_game",
    FatigueLevel.MEDIUM: "riddle",
    FatigueLevel.MEDIUM_HIGH: "brain_teaser",
    FatigueLevel.HIGH: "number_plate"
}

ACTIVITY_PROMPTS = {
    "casual": "Start a friendly conversation with the driver to keep them engaged and awake.",
    "riddle": "Ask the driver a fun riddle and wait for their answer.",
    "story": "Tell the driver a short, interesting story to stimulate their mind.",
    "brain_teaser": "Pose a quick brain teaser or puzzle for the driver.",
    "word_game": "Engage the driver in a word game, like 'I Spy' or '20 Questions'.",
    "math_game": "Challenge the driver with a simple math problem or a mental calculation.",
    "number_plate": "Challenge the driver to spot and remember the numbers on the car in front, or calculate sums or products with the number digits on the licence plate."
}

def get_activity_for_fatigue(fatigue_level):
    return FATIGUE_ACTIVITY_MAP.get(fatigue_level, "casual")

def get_prompt(activity_type, fatigue_level):
    return f"{STARTING_PROMPT} The driver is currently at a {fatigue_level.value} fatigue level. {ACTIVITY_PROMPTS[activity_type]}"

def converse_with_driver(fatigue_level):
    activity_type = get_activity_for_fatigue(fatigue_level)
    # Custom initial prompt to ask for name and destination
    initial_prompt = (
        f"{STARTING_PROMPT}\n"
        "Before we begin, can you tell me your name and where you are heading?"
    )
    print(f"\n--- {activity_type.upper()} for {fatigue_level.value} fatigue ---")
    
    history = [initial_prompt]
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="\n".join(history)
    )
    print("Assistant:", response.text)
    history.append(f"Assistant: {response.text}")
    
    # Get user's name and destination
    user_input = input("\nYou: ")
    history.append(f"Driver: {user_input}")
    # Now start the activity based on fatigue level
    activity_prompt = get_prompt(activity_type, fatigue_level)
    history.append(activity_prompt)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents="\n".join(history)
    )
    print("Assistant:", response.text)
    history.append(f"Assistant: {response.text}")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.strip().lower() == "stop":
            print("Assistant: Goodbye! Stay safe.")
            break
        history.append(f"Driver: {user_input}")
        followup = client.models.generate_content(
            model="gemini-2.0-flash",
            contents="\n".join(history)
        )
        print("Assistant:", followup.text)
        history.append(f"Assistant: {followup.text}")
        
# Example usage:
if __name__ == "__main__":
    # Simulate a fatigue level (replace with actual detection logic)
    fatigue_level = FatigueLevel.HIGH  # Change this for testing
    converse_with_driver(fatigue_level)