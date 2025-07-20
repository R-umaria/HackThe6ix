import pyttsx3
import time

engine = pyttsx3.init()

last_alert_time = 0
alert_cooldown = 10

def speak(text):
    engine.say(text)
    engine.runAndWait()

def get_tier_from_file():
    try:
        with open("tier.txt", "r") as f:
            return f.read().strip().lower()
    except FileNotFoundError:
        return "none"

if __name__ == "__main__":
    while True:
        tier = get_tier_from_file()
        print("Current drowsiness tier:", tier)

        if tier == "high":
            print("Drowsiness HIGH detected, speaking alert...")
            current_time = time.time()
            if current_time - last_alert_time > alert_cooldown:
                speak("Warning! You appear very drowsy. Please take a break.")
                last_alert_time = current_time

        time.sleep(1)
