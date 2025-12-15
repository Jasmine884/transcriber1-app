import tkinter as tk
import threading
import queue
import sounddevice as sd
import numpy as np
import whisper

# ---------------- SETTINGS ----------------
SAMPLE_RATE = 16000
CHANNELS = 2
CHUNK_SECONDS = 2  # smaller chunks = more responsive
STEREO_MIX_DEVICE = 2  # replace with your correct input device index
# -------------------------------------------

audio_queue = queue.Queue()
model = whisper.load_model("base")

# Callback to push audio chunks to the queue
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

# Worker thread to process audio and update the GUI
def transcribe_worker(text_widget, stop_event):
    while not stop_event.is_set():
        try:
            audio = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue  # keep looping
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)  # downmix to mono
        if len(audio) < SAMPLE_RATE // 2:
            continue
        try:
            result = model.transcribe(audio, language="en", fp16=False)
            text = result["text"].strip()
            if text:
                text_widget.insert(tk.END, text + "\n")
                text_widget.see(tk.END)
        except Exception as e:
            print("Transcription error:", e)

# Start transcription
def start_transcription(text_widget):
    global stop_event, stream, thread
    stop_event = threading.Event()
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype="float32",
        blocksize=int(SAMPLE_RATE * CHUNK_SECONDS),
        device=STEREO_MIX_DEVICE,
        callback=audio_callback
    )
    stream.start()
    thread = threading.Thread(target=transcribe_worker, args=(text_widget, stop_event), daemon=True)
    thread.start()

# Stop transcription
def stop_transcription():
    stop_event.set()
    stream.stop()
    stream.close()

# ------------------ GUI --------------------
root = tk.Tk()
root.title("Live Transcriber")

text_box = tk.Text(root, height=20, width=60)
text_box.pack(padx=10, pady=10)

start_btn = tk.Button(root, text="Start Transcription", command=lambda: start_transcription(text_box))
start_btn.pack(pady=5)

stop_btn = tk.Button(root, text="Stop Transcription", command=stop_transcription)
stop_btn.pack(pady=5)

root.mainloop()
