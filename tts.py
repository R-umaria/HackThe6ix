import pyttsx3
import time
import os
import threading
from google import genai
from dotenv import load_dotenv
from globals import DrowsinessTier

# === Load Gemini API key ===
load_dotenv()
GOOGLE_API_KEY = os.getenv("GEMINI_API")
client = genai.Client(api_key=GOOGLE_API_KEY)

# === TTS Setup ===
engine = pyttsx3.init()

# === Fatigue Logic ===
def get_tier_from_file():
    try:
        with open("tier.txt", "r") as f:
            return f.read().strip().lower()
    except FileNotFoundError:
        return "none"

# === Gemini Assistant Logic ===
STARTING_PROMPT = """
You are Alex, a helpful assistant designed to keep a driver engaged and
alert during long drives. Based on the driver's fatigue level, suggest
an activity that is appropriate for their current state. The activities
should be engaging, stimulating, and help maintain focus on the road.
YOU will interact with the driver in a friendly and supportive manner.
Ask for their name and destination and adapt your style to their alertness level.
"""

FATIGUE_ACTIVITY_MAP = {
    DrowsinessTier.NONE: "casual",
    DrowsinessTier.LOW: "word_game",
    DrowsinessTier.MEDIUM_LOW: "math_game",
    DrowsinessTier.MEDIUM: "riddle",
    DrowsinessTier.MEDIUM_HIGH: "brain_teaser",
    DrowsinessTier.HIGH: "number_plate"
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

# === Conversational Assistant ===
class AlexAssistant:
    def __init__(self):
        self.history = []
        self.active = False
        self.thread = None
        self.current_tier = DrowsinessTier.NONE
        self.is_speaking = False  # To block listening while speaking

    def speak(self, text):
        """ Speak the text and block until it's finished speaking """
        print("[Alex]", text)
        self.is_speaking = True
        engine.say(text)
        engine.runAndWait()  # Block until speech is done
        self.is_speaking = False  # Reset after speaking is done

    def start(self):
        if self.active:
            return
        self.active = True
        self.thread = threading.Thread(target=self._run)
        self.thread.start()

    def _run(self):
        """ Start the assistant's conversation loop """
        self.current_tier = self._get_drowsiness_tier()  # Get fatigue level before interacting
        activity = get_activity_for_fatigue(self.current_tier)

        # Adjust prompt based on the current fatigue level
        self.history = [
            f"{STARTING_PROMPT}\nBefore we begin, can you tell me your name and destination?"
        ]
        
        # Call Gemini to get the initial response based on the fatigue level
        prompt = get_prompt(activity, self.current_tier)
        res = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        self.speak(res.text)
        self.history.append(f"Assistant: {res.text}")

        # Continuously interact with the user in a loop
        while self.active:
            try:
                # Wait until speaking is finished to start "listening" (simulated input)
                if not self.is_speaking:
                    user_input = self.simulate_user_input()  # Simulate user input (no audio)
                    if user_input:
                        self.history.append(f"Driver: {user_input}")
                        res = client.models.generate_content(
                            model="gemini-2.0-flash",
                            contents="\n".join(self.history)
                        )
                        self.speak(res.text)
                        self.history.append(f"Assistant: {res.text}")
                    else:
                        self.speak("Sorry, I didn't get that. Can you say it again?")
                else:
                    time.sleep(0.5)  # Wait until speaking is finished before checking again
                    
            except KeyboardInterrupt:
                self.active = False
                break

    def simulate_user_input(self):
        """ Simulate text input from the user (no actual speech input) """
        user_input = input("[Driver] Please type your response: ")
        return user_input.strip() if user_input else None

    def _get_drowsiness_tier(self):
        """ Read fatigue level from file """
        level = get_tier_from_file()
        try:
            return DrowsinessTier(level.upper())
        except ValueError:
            return DrowsinessTier.NONE

    def stop(self):
        """ Stop the assistant """
        self.active = False
        if self.thread:
            self.thread.join()
            self.thread = None

# === Main run loop ===
if __name__ == "__main__":
    alex = AlexAssistant()
    alex.start()  # Always on, behavior changes with tier
