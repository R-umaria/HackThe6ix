import pyttsx3

engine = pyttsx3.init()
engine.setProperty('rate', 160)  # slower for clarity

def speak_text(text):
    print("Speaking:", text)
    engine.say(text)
    engine.runAndWait()
