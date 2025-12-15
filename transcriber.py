import os
import sounddevice as sd
import numpy as np
import whisper
import queue
import threading
os.system("cls")
# -------- SETTINGS --------
SAMPLE_RATE = 16000
CHANNELS = 2               # Stereo Mix is stereo
CHUNK_SECONDS = 5

# Find your Stereo Mix index from list_devices.py
STEREO_MIX_DEVICE = 2
# --------------------------

audio_queue = queue.Queue()

print("Loading Whisper model (this may take a minute)...")
model = whisper.load_model("base")
print("Model loaded.\n")

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

def transcribe_worker():
    while True:
        audio = audio_queue.get()
        
        # downmix to mono for Whisper
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        if len(audio) < SAMPLE_RATE:
            continue

        try:
            result = model.transcribe(
                audio,
                language="en",
                fp16=False
            )
            text = result["text"].strip()
            if text:
                print(">>", text)
        except Exception as e:
            print("Transcription error:", e)

def main():
    print("Listening to SYSTEM AUDIO via Stereo Mix...")
    print("Play any audio. Press Ctrl+C to stop.\n")

    threading.Thread(target=transcribe_worker, daemon=True).start()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * CHUNK_SECONDS),
        device=STEREO_MIX_DEVICE,
        callback=audio_callback
    ):
        while True:
            sd.sleep(1000)

if __name__ == "__main__":
    main()
