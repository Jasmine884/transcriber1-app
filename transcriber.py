import tkinter as tk
from tkinter import ttk, messagebox
import threading
import queue
import sounddevice as sd
import numpy as np
import whisper
import time

# ---------------- SETTINGS ----------------
SAMPLE_RATE = 16000
DEFAULT_WINDOW_SECONDS = 10       # how much recent audio to transcribe each pass
DEFAULT_TRANSCRIBE_EVERY = 3.0     # seconds between transcriptions
MODEL_NAME = "base"
LANGUAGE = "en"
# -----------------------------------------

audio_queue = queue.Queue()
model = whisper.load_model(MODEL_NAME)

def list_input_devices():
    devices = sd.query_devices()
    inputs = []
    for i, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            inputs.append((i, d["name"], d["max_input_channels"]))
    return inputs

def safe_gui_insert(root, text_widget, text):
    # Tkinter must be updated on the main thread
    def _do():
        text_widget.insert(tk.END, text + "\n")
        text_widget.see(tk.END)
    root.after(0, _do)

def audio_callback(indata, frames, time_info, status):
    if status:
        # Avoid printing too much; but keep for debugging
        print("Audio status:", status)
    # Copy to avoid referencing the same buffer
    audio_queue.put(in_
