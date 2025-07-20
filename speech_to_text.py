import speech_recognition as sr

recognizer = sr.Recognizer()

def get_driver_input():
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
        try:
            return recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print("Speech recognition error:", e)
            return None
