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
    audio_queue.put(indata.copy())

def transcribe_worker(root, text_widget, stop_event, window_seconds, transcribe_every):
    """
    Buffers audio continuously and transcribes the most recent `window_seconds`
    every `transcribe_every` seconds.
    """
    buffer = np.zeros(int(SAMPLE_RATE * window_seconds), dtype=np.float32)
    last_transcribe_time = 0.0
    last_text = ""  # basic de-dup to reduce repeated output

    while not stop_event.is_set():
        try:
            chunk = audio_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        # chunk shape: (frames, channels) or (frames,)
        if chunk.ndim == 2:
            # downmix to mono
            chunk = chunk.mean(axis=1)

        chunk = chunk.astype(np.float32, copy=False)

        # Resample guard: sounddevice should already provide correct samplerate,
        # but if device doesn't, you'll get wrong speed. Keep it simple here.
        # (If you need resampling, we can add it cleanly.)

        # Roll buffer and append new audio
        n = len(chunk)
        if n >= len(buffer):
            buffer[:] = chunk[-len(buffer):]
        else:
            buffer = np.roll(buffer, -n)
            buffer[-n:] = chunk

        now = time.time()
        if now - last_transcribe_time < transcribe_every:
            continue
        last_transcribe_time = now

        # Skip if mostly silence (very cheap VAD-ish gate)
        if np.max(np.abs(buffer)) < 0.01:
            continue

        try:
            # Whisper expects float32 mono at 16k
            result = model.transcribe(
                buffer,
                language=LANGUAGE,
                fp16=False
            )
            text = (result.get("text") or "").strip()

            # avoid spamming repeats
            if text and text != last_text:
                last_text = text
                safe_gui_insert(root, text_widget, text)

        except Exception as e:
            print("Transcription error:", e)

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Live Transcriber")

        self.stop_event = None
        self.stream = None
        self.thread = None

        self.devices = list_input_devices()

        # ---- UI ----
        top = ttk.Frame(root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Input device:").pack(side="left")

        self.device_var = tk.StringVar()
        device_names = [f"{idx}: {name}" for idx, name, _ch in self.devices]
        if device_names:
            self.device_var.set(device_names[0])
        self.device_menu = ttk.Combobox(top, textvariable=self.device_var, values=device_names, width=60, state="readonly")
        self.device_menu.pack(side="left", padx=8)

        settings = ttk.Frame(root, padding=(10, 0, 10, 10))
        settings.pack(fill="x")

        ttk.Label(settings, text="Window (sec):").grid(row=0, column=0, sticky="w")
        self.window_var = tk.IntVar(value=DEFAULT_WINDOW_SECONDS)
        ttk.Spinbox(settings, from_=5, to=30, textvariable=self.window_var, width=5).grid(row=0, column=1, sticky="w", padx=6)

        ttk.Label(settings, text="Transcribe every (sec):").grid(row=0, column=2, sticky="w", padx=(20, 0))
        self.every_var = tk.DoubleVar(value=DEFAULT_TRANSCRIBE_EVERY)
        ttk.Spinbox(settings, from_=1, to=10, increment=0.5, textvariable=self.every_var, width=5).grid(row=0, column=3, sticky="w", padx=6)

        self.text_box = tk.Text(root, height=20, width=80)
        self.text_box.pack(padx=10, pady=10)

        btns = ttk.Frame(root, padding=(10, 0, 10, 10))
        btns.pack(fill="x")

        self.start_btn = ttk.Button(btns, text="Start", command=self.start)
        self.start_btn.pack(side="left")

        self.stop_btn = ttk.Button(btns, text="Stop", command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=8)

        self.clear_btn = ttk.Button(btns, text="Clear", command=lambda: self.text_box.delete("1.0", tk.END))
        self.clear_btn.pack(side="left", padx=8)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        if not self.devices:
            messagebox.showerror("No input devices", "No audio input devices found.")
            self.start_btn.config(state="disabled")

    def _selected_device_index(self):
        s = self.device_var.get()
        if ":" not in s:
            return None
        try:
            return int(s.split(":")[0].strip())
        except ValueError:
            return None

    def start(self):
        if self.stream is not None:
            return

        device_idx = self._selected_device_index()
        if device_idx is None:
            messagebox.showerror("Device error", "Please select a valid input device.")
            return

        window_seconds = int(self.window_var.get())
        transcribe_every = float(self.every_var.get())

        self.stop_event = threading.Event()

        # Use 1 channel to avoid stereo device weirdness; we downmix anyway if needed.
        try:
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=device_idx,
                callback=audio_callback,
            )
            self.stream.start()
        except Exception as e:
            self.stream = None
            messagebox.showerror("Audio error", f"Could not start audio stream:\n{e}")
            return

        self.thread = threading.Thread(
            target=transcribe_worker,
            args=(self.root, self.text_box, self.stop_event, window_seconds, transcribe_every),
            daemon=True
        )
        self.thread.start()

        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        safe_gui_insert(self.root, self.text_box, "[Started transcription]")

    def stop(self):
        if self.stream is None:
            return

        if self.stop_event:
            self.stop_event.set()

        try:
            self.stream.stop()
            self.stream.close()
        except Exception:
            pass

        self.stream = None
        self.thread = None
        self.stop_event = None

        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        safe_gui_insert(self.root, self.text_box, "[Stopped transcription]")

    def on_close(self):
        self.stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
