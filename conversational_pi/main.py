# main.py

from stt.whisper_runner import transcribe_audio
from tts.tts_engine import speak_text
from gemini.gemini_api import ask_gemini

if __name__ == "__main__":
    print("Alex is listening... (Ctrl+C to stop)")
    try:
        while True:
            user_input = transcribe_audio()
            print("Transcript:", repr(user_input))  # debug aid
            if not user_input.strip():
                print("Nothing heard.")
                continue
            response = ask_gemini(user_input)
            print("You said:", user_input)
            speak_text(response)
    except KeyboardInterrupt:
        print("Goodbye!")
