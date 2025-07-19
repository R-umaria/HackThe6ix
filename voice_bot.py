# filename: voice_bot.py
import speech_recognition as sr
import pyttsx3

recognizer = sr.Recognizer()
tts = pyttsx3.init()

def speak(text):
    print(f"Bot: {text}")
    tts.say(text)
    tts.runAndWait()

def listen():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio) #type: ignore
        except sr.UnknownValueError:
            return "Sorry, I didn't catch that."
        except sr.RequestError:
            return "Speech service down."

while True:
    user_input = listen()
    print(f"You said: {user_input}")
    if "stop" in user_input.lower():
        speak("Goodbye!")
        break
    else:
        speak("I heard you say " + user_input)
