"""
gui.py — Tkinter source-selection window.

One public function:
  show_source_chooser() → returns "webcam", "video", or None
"""

import os
import tkinter as tk
from tkinter import messagebox

from config import SAMPLE_VIDEO


def show_source_chooser():
    """
    Display a small GUI window with two buttons so the user can pick
    between the live webcam and the sample squat video.

    Returns
    -------
    "webcam"  — user clicked Use Webcam
    "video"   — user clicked Use Sample Video (and the file exists)
    None      — user closed the window without choosing
    """
    choice = {"value": None}   # dict so inner callbacks can write to it

    root = tk.Tk()
    root.title("Select Input Source")
    root.resizable(False, False)
    root.eval("tk::PlaceWindow . center")
    root.attributes("-topmost", True)

    tk.Label(root, text="MediaPipe Pose — FYP Demo",
             font=("Helvetica", 13, "bold"), pady=10).pack(padx=30)
    tk.Label(root, text="Choose a video source to begin:",
             font=("Helvetica", 10), pady=4).pack()

    def on_webcam():
        choice["value"] = "webcam"
        root.destroy()

    def on_video():
        if not os.path.exists(SAMPLE_VIDEO):
            messagebox.showerror(
                "File Not Found",
                f"Sample video not found:\n{os.path.abspath(SAMPLE_VIDEO)}\n\n"
                "Place the video file in the assets/ folder.",
            )
            return   # keep the chooser open so the user can pick webcam instead
        choice["value"] = "video"
        root.destroy()

    btn = {"width": 22, "pady": 6, "font": ("Helvetica", 11)}
    tk.Button(root, text="Use Webcam",       command=on_webcam, **btn).pack(pady=(12, 4),  padx=30)
    tk.Button(root, text="Use Sample Video", command=on_video,  **btn).pack(pady=(4,  16), padx=30)

    root.mainloop()
    return choice["value"]
