"""
Tkinter GUI for cropping the tactile capture in this directory.

Run:
    python crop_gui.py

What you get
------------
A single window with:

  * a small RGB thumbnail at the START of the current range
  * a small RGB thumbnail at the END   of the current range
  * a force-strip plot of total-force across the whole capture, with the
    selected range highlighted (lets you eyeball where contact happens)
  * two sliders to set the start and end frame
  * a "Frame" / "Time" readout next to each
  * a "Dry run" button (prints what would happen without writing)
  * a "Crop!" button (calls crop.py internally and writes the cropped dataset)

The window is single-file Tkinter — no extra dependencies beyond Pillow,
which is already available in the lerobot env.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
import pandas as pd
from PIL import Image, ImageTk

THUMB_SIZE = (320, 240)
STRIP_SIZE = (640, 100)


def load_df(root: Path) -> pd.DataFrame:
    parquet = root / "frames.parquet"
    if not parquet.exists():
        raise SystemExit(f"frames.parquet not found in {root}")
    df = pd.read_parquet(parquet).sort_values("frame_idx").reset_index(drop=True)
    return df


def total_force_per_frame(df: pd.DataFrame) -> np.ndarray:
    """Sum of all calibrated force_N samples across both hands and all 5 fingers."""
    cols = [f"{side}_{f}_force_N_cal"
            for side in ("lh", "rh")
            for f in ("thumb", "index", "middle", "ring", "little")]
    series = []
    for c in cols:
        # Each entry is an array of 12 values; sum along that
        series.append(np.array([np.asarray(v).sum() for v in df[c].values], dtype=np.float32))
    total = np.sum(series, axis=0)
    return total


def render_force_strip(force: np.ndarray, start_idx: int, end_idx: int,
                       w=STRIP_SIZE[0], h=STRIP_SIZE[1]) -> Image.Image:
    """Draw the force strip with the selected range highlighted as a PIL image."""
    img = Image.new("RGB", (w, h), (22, 22, 28))
    pixels = np.array(img)

    n = len(force)
    if n == 0:
        return img
    fmax = max(float(force.max()), 1e-3)

    # x positions for each frame
    xs = (np.arange(n) / max(n - 1, 1) * (w - 1)).astype(int)
    ys = (h - 4 - (force / fmax) * (h - 12)).astype(int)

    # highlight band: column-wise tint between start and end
    sx = int(start_idx / max(n - 1, 1) * (w - 1))
    ex = int(end_idx   / max(n - 1, 1) * (w - 1))
    pixels[:, sx:ex + 1] = (40, 60, 90)  # bluish highlight

    # plot the curve as a polyline
    for i in range(n - 1):
        x0, y0 = xs[i], ys[i]
        x1, y1 = xs[i + 1], ys[i + 1]
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for s in range(steps + 1):
            t = s / steps
            x = int(x0 + (x1 - x0) * t)
            y = int(y0 + (y1 - y0) * t)
            if 0 <= y < h and 0 <= x < w:
                pixels[y, x] = (255, 200, 80)

    # start / end markers (vertical lines)
    for x, col in ((sx, (90, 220, 255)), (ex, (255, 120, 120))):
        if 0 <= x < w:
            pixels[:, x] = col

    return Image.fromarray(pixels)


class CropGUI:
    def __init__(self, root: tk.Tk, data_dir: Path):
        self.root = root
        self.root.title(f"Tactile crop — {data_dir.name}")
        self.root.configure(bg="#15151a")

        self.data_dir = data_dir
        self.df = load_df(data_dir)
        self.n_frames = len(self.df)
        self.t0 = float(self.df["timestamp"].iloc[0])
        self.duration = float(self.df["timestamp"].iloc[-1]) - self.t0
        self.force = total_force_per_frame(self.df)

        self._build_ui()
        self._refresh()

    # ----- UI ---------------------------------------------------------------

    def _build_ui(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TLabel", background="#15151a", foreground="#e6e6ee", font=("Inter", 11))
        style.configure("Title.TLabel", font=("Inter", 13, "bold"))
        style.configure("TButton", padding=6)
        style.configure("Horizontal.TScale", background="#15151a")

        # top: thumbnails
        top = tk.Frame(self.root, bg="#15151a")
        top.pack(padx=14, pady=(14, 6))

        self.start_img_lbl = tk.Label(top, bg="#22222a")
        self.start_img_lbl.grid(row=0, column=0, padx=6)
        self.end_img_lbl = tk.Label(top, bg="#22222a")
        self.end_img_lbl.grid(row=0, column=1, padx=6)

        ttk.Label(top, text="START", style="Title.TLabel").grid(row=1, column=0)
        ttk.Label(top, text="END",   style="Title.TLabel").grid(row=1, column=1)

        # readouts
        self.start_lbl = ttk.Label(top, text="")
        self.start_lbl.grid(row=2, column=0)
        self.end_lbl = ttk.Label(top, text="")
        self.end_lbl.grid(row=2, column=1)

        # mid: force strip
        strip_frame = tk.Frame(self.root, bg="#15151a")
        strip_frame.pack(padx=14, pady=(8, 4))
        self.strip_lbl = tk.Label(strip_frame, bg="#15151a")
        self.strip_lbl.pack()
        ttk.Label(strip_frame, text="Total force across capture  ·  blue = start  ·  red = end",
                  style="TLabel").pack()

        # sliders
        sliders = tk.Frame(self.root, bg="#15151a")
        sliders.pack(padx=14, pady=(4, 4), fill="x")

        self.start_var = tk.IntVar(value=0)
        self.end_var = tk.IntVar(value=self.n_frames - 1)

        ttk.Label(sliders, text="Start frame").grid(row=0, column=0, sticky="w")
        self.start_scale = ttk.Scale(sliders, from_=0, to=self.n_frames - 1,
                                     orient="horizontal",
                                     command=lambda v: self._on_start(int(float(v))),
                                     length=560)
        self.start_scale.grid(row=0, column=1, padx=8, pady=4, sticky="ew")
        self.start_scale.set(0)

        ttk.Label(sliders, text="End frame").grid(row=1, column=0, sticky="w")
        self.end_scale = ttk.Scale(sliders, from_=0, to=self.n_frames - 1,
                                   orient="horizontal",
                                   command=lambda v: self._on_end(int(float(v))),
                                   length=560)
        self.end_scale.grid(row=1, column=1, padx=8, pady=4, sticky="ew")
        self.end_scale.set(self.n_frames - 1)
        sliders.grid_columnconfigure(1, weight=1)

        # buttons
        btns = tk.Frame(self.root, bg="#15151a")
        btns.pack(padx=14, pady=(8, 14))
        ttk.Button(btns, text="Dry run", command=self._dry_run).grid(row=0, column=0, padx=6)
        ttk.Button(btns, text="Crop!",   command=self._crop).grid(row=0, column=1, padx=6)
        ttk.Button(btns, text="Reset",   command=self._reset).grid(row=0, column=2, padx=6)
        ttk.Button(btns, text="Quit",    command=self.root.destroy).grid(row=0, column=3, padx=6)

        # status line
        self.status = ttk.Label(self.root, text="", style="TLabel", wraplength=640, justify="left")
        self.status.pack(padx=14, pady=(0, 12), fill="x")

    # ----- slider callbacks -------------------------------------------------

    def _on_start(self, v: int):
        if v >= self.end_var.get():
            v = max(0, self.end_var.get() - 1)
            self.start_scale.set(v)
        self.start_var.set(v)
        self._refresh()

    def _on_end(self, v: int):
        if v <= self.start_var.get():
            v = min(self.n_frames - 1, self.start_var.get() + 1)
            self.end_scale.set(v)
        self.end_var.set(v)
        self._refresh()

    def _reset(self):
        self.start_scale.set(0)
        self.end_scale.set(self.n_frames - 1)
        self.start_var.set(0)
        self.end_var.set(self.n_frames - 1)
        self._refresh()

    # ----- refresh ----------------------------------------------------------

    def _load_thumb(self, idx: int) -> ImageTk.PhotoImage:
        path = self.data_dir / self.df.iloc[idx]["rgb_path"]
        if not path.exists():
            img = Image.new("RGB", THUMB_SIZE, (60, 60, 70))
        else:
            img = Image.open(path).convert("RGB")
            img.thumbnail(THUMB_SIZE)
            # pad to fixed size so layout doesn't jump
            canvas = Image.new("RGB", THUMB_SIZE, (15, 15, 20))
            ox = (THUMB_SIZE[0] - img.size[0]) // 2
            oy = (THUMB_SIZE[1] - img.size[1]) // 2
            canvas.paste(img, (ox, oy))
            img = canvas
        return ImageTk.PhotoImage(img)

    def _refresh(self):
        if not hasattr(self, "status"):
            return  # widgets not all built yet
        s = self.start_var.get()
        e = self.end_var.get()

        # thumbnails (keep refs to avoid GC)
        self._start_photo = self._load_thumb(s)
        self._end_photo = self._load_thumb(e)
        self.start_img_lbl.configure(image=self._start_photo)
        self.end_img_lbl.configure(image=self._end_photo)

        ts_s = float(self.df.iloc[s]["timestamp"]) - self.t0
        ts_e = float(self.df.iloc[e]["timestamp"]) - self.t0
        self.start_lbl.configure(text=f"frame {s}    t = {ts_s:6.2f} s")
        self.end_lbl.configure(text=  f"frame {e}    t = {ts_e:6.2f} s")

        # force strip
        strip = render_force_strip(self.force, s, e)
        self._strip_photo = ImageTk.PhotoImage(strip)
        self.strip_lbl.configure(image=self._strip_photo)

        self.status.configure(text=f"Will keep {e - s + 1} frames "
                                   f"({ts_e - ts_s:.2f} s of "
                                   f"{self.duration:.2f} s).")

    # ----- actions ----------------------------------------------------------

    def _run_crop(self, dry: bool):
        s = self.start_var.get()
        e = self.end_var.get()
        cmd = [sys.executable, "-m", "tactile_reader.crop",
               str(self.data_dir), "--start", str(s), "--end", str(e)]
        if dry:
            cmd.append("--dry-run")
        try:
            out = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.status.configure(text=out.stdout.strip() or "(no output)")
        except subprocess.CalledProcessError as ex:
            messagebox.showerror("crop.py failed",
                                 f"stdout:\n{ex.stdout}\n\nstderr:\n{ex.stderr}")
            self.status.configure(text=f"FAILED: {ex.stderr.strip()}")

    def _dry_run(self):
        self._run_crop(dry=True)

    def _crop(self):
        s = self.start_var.get()
        e = self.end_var.get()
        if not messagebox.askyesno("Confirm crop",
                                   f"Crop to frames [{s}, {e}] "
                                   f"({e - s + 1} frames) and write a new dataset?"):
            return
        self._run_crop(dry=False)


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="tactile-reader-crop-gui",
        description="GUI for cropping a tactile capture directory by frame range.")
    ap.add_argument("data_dir", nargs="?", default=".",
                    help="capture directory (default: current working dir)")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir).resolve()
    if not (data_dir / "frames.parquet").exists():
        raise SystemExit(f"frames.parquet not found in {data_dir}. "
                         f"Usage: tactile-reader-crop-gui <dir>")

    root = tk.Tk()
    try:
        root.tk_setPalette(background="#15151a", foreground="#e6e6ee",
                           activeBackground="#2a2a35", activeForeground="#ffffff")
    except Exception:
        pass
    CropGUI(root, data_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
