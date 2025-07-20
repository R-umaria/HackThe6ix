import whisper
import sounddevice as sd
import numpy as np
import tempfile
import scipy.io.wavfile as wav

model = whisper.load_model("base")

def record_audio(duration=5, samplerate=16000):
    print("Listening...")
    audio = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1)
    sd.wait()
    return audio.flatten()

def transcribe_audio():
    audio = record_audio()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav.write(f.name, 16000, (audio * 32767).astype(np.int16))
        result = model.transcribe(f.name)
        return result["text"]
