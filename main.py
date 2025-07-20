import cv2
import time
import threading
from face_recognition import *  # Import your face recognition and drowsiness detection functions
from gemini import *  # Import your Gemini conversation handling
import speech_recognition as sr  # For speech-to-text
import pyttsx3  # For text-to-speech
import globals

# Initialize the speech recognition and text-to-speech
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Initialize the webcam feed
cap = cv2.VideoCapture(0)

# Function for Speech-to-Text
def listen_for_driver_input():
    while True:
        with sr.Microphone() as source:
            print("Listening for driver's input...")
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio)
                print(f"Driver said: {text}")
                return text
            except sr.UnknownValueError:
                print("Could not understand audio, trying again...")
            except sr.RequestError as e:
                print(f"Error with the speech service; {e}")
                break

# Function for Text-to-Speech
def speak_to_driver(text):
    print(f"Assistant says: {text}")
    engine.say(text)
    engine.runAndWait()

# Function to monitor driver fatigue and activate Gemini if needed
def monitor_driver_fatigue():
    global fatigue_level

    start_time = time.time()

    with FaceLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Face landmark detection and drowsiness level update
            results = process_face_landmarks(frame, landmarker, start_time)  # Make sure this is implemented in face_recognition.py
            fatigue_level = get_fatigue_level_from_drowsiness(results)  # Make sure to return a fatigue level (use globals.FatigueLevel)
            
            if fatigue_level in [FatigueLevel.MEDIUM, FatigueLevel.MEDIUM_HIGH, FatigueLevel.HIGH]:
                print(f"Driver is {fatigue_level.value}, activating Gemini.")
                converse_with_driver(fatigue_level)  # Activate Gemini when drowsiness level is high enough

            # Display drowsiness level on frame (optional)
            cv2.putText(frame, f"Drowsiness: {fatigue_level.value}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Drowsiness Detection", frame)

            # If the user presses ESC, exit
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

# Main function
def main():
    # Start the fatigue monitoring in a separate thread
    fatigue_thread = threading.Thread(target=monitor_driver_fatigue)
    fatigue_thread.start()

    # Listen for driver input concurrently
    while True:
        driver_input = listen_for_driver_input()  # Wait for driver input
        if driver_input.lower() == "stop":
            speak_to_driver("Goodbye! Stay safe.")
            break
        else:
            speak_to_driver(f"You said: {driver_input}")

if __name__ == "__main__":
    main()
