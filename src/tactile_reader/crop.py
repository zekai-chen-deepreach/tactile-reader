"""
Crop the capture in this directory to a frame range [start, end] (inclusive),
producing a new self-contained dataset directory.

Usage
-----
1. Open the rerun viewer (`python visualize.py`), scrub the timeline, and read
   the `frame_idx` of the first and last frame you want to keep from the time
   panel.
2. Run:
       python crop.py --start 24 --end 287
   or, using fractional positions of the recording:
       python crop.py --start-pct 0.10 --end-pct 0.90
   or, using rerun-style timestamps (Unix seconds shown in the viewer):
       python crop.py --start-ts 1779492540.0 --end-ts 1779492560.0

By default writes to `./052211_cropped/` next to the source.  Pass `--out` to
override.  The source directory is never modified.

What it produces
----------------
out/
  frames.parquet     subset rows, with `frame_idx` re-indexed from 0
  oak_imu.parquet    rows with original timestamp in [start_ts, end_ts]
  rgb/000000.jpg ... renamed contiguously
  mono_left/000000.png ...
  mono_right/000000.png ...
  calib.json         copy of source
  crop_meta.json     {source, start_frame, end_frame, start_ts, end_ts, n_frames}
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        prog="tactile-reader-crop",
        description="Crop a tactile capture directory to a frame range.")
    ap.add_argument("data_dir", nargs="?", default=".",
                    help="source capture directory (default: current working dir)")
    ap.add_argument("--out", default=None,
                    help="output directory (default: <data_dir>_cropped, sibling of source)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--start", type=int, help="start frame_idx (inclusive)")
    g.add_argument("--start-pct", type=float, help="start as fraction in [0,1]")
    g.add_argument("--start-ts", type=float, help="start time in SECONDS FROM START (matches the rerun viewer's timeline, 0..duration)")
    g2 = ap.add_mutually_exclusive_group(required=True)
    g2.add_argument("--end", type=int, help="end frame_idx (inclusive)")
    g2.add_argument("--end-pct", type=float, help="end as fraction in [0,1]")
    g2.add_argument("--end-ts", type=float, help="end time in SECONDS FROM START (matches the rerun viewer's timeline)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be done without writing anything")
    ap.add_argument("--no-images", action="store_true",
                    help="do not copy RGB/mono images (parquet still trimmed)")
    return ap.parse_args(argv)


def resolve_bound(df: pd.DataFrame, frame: int | None, pct: float | None,
                  ts: float | None, label: str) -> int:
    """Convert one of --{label}/--{label}-pct/--{label}-ts to a frame_idx."""
    if frame is not None:
        return int(frame)
    if pct is not None:
        assert 0.0 <= pct <= 1.0, f"--{label}-pct must be in [0,1]"
        return int(round(pct * (len(df) - 1)))
    if ts is not None:
        # ts is in seconds-from-start (to match the rerun viewer's timeline)
        target = df["timestamp"].min() + ts
        diff = (df["timestamp"] - target).abs()
        return int(df.loc[diff.idxmin(), "frame_idx"])
    raise ValueError(f"no bound provided for {label}")


def main(argv=None):
    args = parse_args(argv)
    src = Path(args.data_dir).resolve()
    if not (src / "frames.parquet").exists():
        raise SystemExit(f"frames.parquet not found in {src}. "
                         f"Pass the path to a capture directory: tactile-reader-crop <dir> --start ... --end ...")

    df = pd.read_parquet(src / "frames.parquet")
    df = df.sort_values("frame_idx").reset_index(drop=True)
    n = len(df)
    print(f"source has {n} frames, frame_idx in "
          f"[{df['frame_idx'].min()}, {df['frame_idx'].max()}], "
          f"ts span {df['timestamp'].max() - df['timestamp'].min():.2f}s")

    s_frame = resolve_bound(df, args.start, getattr(args, "start_pct"),
                             getattr(args, "start_ts"), "start")
    e_frame = resolve_bound(df, args.end, getattr(args, "end_pct"),
                             getattr(args, "end_ts"), "end")
    if s_frame > e_frame:
        s_frame, e_frame = e_frame, s_frame

    mask = (df["frame_idx"] >= s_frame) & (df["frame_idx"] <= e_frame)
    sub = df[mask].reset_index(drop=True)
    if len(sub) == 0:
        raise SystemExit(f"empty crop: no frames in [{s_frame}, {e_frame}]")

    start_ts = float(sub["timestamp"].iloc[0])
    end_ts = float(sub["timestamp"].iloc[-1])
    print(f"keeping {len(sub)} frames: frame_idx [{s_frame}, {e_frame}], "
          f"duration {end_ts - start_ts:.2f}s")

    if args.out is None:
        out = src.parent / f"{src.name}_cropped"
    else:
        out = Path(args.out).resolve()

    if args.dry_run:
        print(f"[dry-run] would write to {out}")
        return

    if out.exists():
        # Be polite: don't blow away an existing directory silently. Append a counter.
        i = 2
        while (out.parent / f"{out.name}_{i}").exists():
            i += 1
        out = out.parent / f"{out.name}_{i}"
        print(f"output exists; using {out} instead")
    out.mkdir(parents=True, exist_ok=False)
    (out / "rgb").mkdir(exist_ok=True)
    (out / "mono_left").mkdir(exist_ok=True)
    (out / "mono_right").mkdir(exist_ok=True)

    # Reindex frame_idx, rewrite *_path columns to point at the new contiguous filenames.
    new_rows = []
    for new_idx, (_, row) in enumerate(sub.iterrows()):
        new_row = row.copy()
        new_row["frame_idx"] = new_idx
        new_row["rgb_path"] = f"rgb/{new_idx:06d}.jpg"
        new_row["mono_left_path"] = f"mono_left/{new_idx:06d}.png"
        new_row["mono_right_path"] = f"mono_right/{new_idx:06d}.png"
        new_rows.append(new_row)

        if not args.no_images:
            for col, dst_dir, dst_name in (
                ("rgb_path", "rgb", f"{new_idx:06d}.jpg"),
                ("mono_left_path", "mono_left", f"{new_idx:06d}.png"),
                ("mono_right_path", "mono_right", f"{new_idx:06d}.png"),
            ):
                p = src / row[col]
                if p.exists():
                    shutil.copy2(p, out / dst_dir / dst_name)

    new_df = pd.DataFrame(new_rows)
    new_df.to_parquet(out / "frames.parquet", index=False)

    # OAK IMU: filter by timestamp window. Note frames.parquet timestamps are
    # Unix seconds while oak_imu.parquet timestamps appear to be a different
    # clock (host monotonic), so we filter by relative position within the
    # original capture to be robust.
    if (src / "oak_imu.parquet").exists():
        imu = pd.read_parquet(src / "oak_imu.parquet")
        full_start = float(df["timestamp"].iloc[0])
        full_end = float(df["timestamp"].iloc[-1])
        rel_start = (start_ts - full_start) / max(full_end - full_start, 1e-9)
        rel_end = (end_ts - full_start) / max(full_end - full_start, 1e-9)
        imu_t0 = float(imu["timestamp"].iloc[0])
        imu_t1 = float(imu["timestamp"].iloc[-1])
        imu_start = imu_t0 + rel_start * (imu_t1 - imu_t0)
        imu_end = imu_t0 + rel_end * (imu_t1 - imu_t0)
        imu_sub = imu[(imu["timestamp"] >= imu_start) & (imu["timestamp"] <= imu_end)].reset_index(drop=True)
        imu_sub.to_parquet(out / "oak_imu.parquet", index=False)
        print(f"oak_imu: kept {len(imu_sub)} / {len(imu)} samples")

    if (src / "calib.json").exists():
        shutil.copy2(src / "calib.json", out / "calib.json")

    meta = {
        "source": str(src),
        "start_frame": int(s_frame),
        "end_frame": int(e_frame),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "n_frames": len(sub),
        "duration_sec": end_ts - start_ts,
    }
    (out / "crop_meta.json").write_text(json.dumps(meta, indent=2))

    print(f"\nWrote crop to: {out}")
    print(f"  frames.parquet: {len(sub)} rows")
    if not args.no_images:
        print(f"  rgb/, mono_left/, mono_right/: {len(sub)} files each")
    print(f"  crop_meta.json: {meta}")


if __name__ == "__main__":
    main()
