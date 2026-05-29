# tactile-reader

Rerun-based visualizer and timeline cropper for tactile-glove + OAK camera captures.

Built for the DeepReach data-collection setup: each capture is a directory containing
`frames.parquet` (per-frame tactile + IMU rows), `oak_imu.parquet` (high-rate IMU),
`rgb/`, `mono_left/`, `mono_right/` image folders, and a `calib.json` with finger-bend
and tactile-zero calibration.

## What you get

`tactile-reader-viz <dir>`
&nbsp;&nbsp;Opens a Rerun viewer with:
- the RGB camera stream (top center)
- two large hand views (left/right) with:
  - skin-tone hand silhouette
  - inferno tactile heatfield with bloom halo
  - white iso-pressure contour lines
  - 120 individual taxel points (size + color scaled by pressure)
  - 5 fingertip force-vector arrows with per-finger Newton readouts
  - 5 bend-angle bars at the top
  - inset pressure colorbar
- 3 time-series panels: per-finger force (N), per-finger bend angle, OAK IMU
- a scrubbable 0..duration timeline shared across everything

`tactile-reader-crop <dir> --start S --end E`
&nbsp;&nbsp;Writes a new self-contained dataset directory cropped to frames [S, E]:
re-indexed `frames.parquet`, renamed contiguous RGB/mono image files, IMU subset,
`calib.json` copy, and a `crop_meta.json` recording the cut.

`tactile-reader-crop-gui <dir>`
&nbsp;&nbsp;Tkinter GUI: two slider handles (start / end), live RGB thumbnails,
a force-strip plot showing where contact happens, and a Crop button that drives
`tactile-reader-crop` under the hood.

## Install

```bash
git clone https://github.com/zekai-chen-deepreach/tactile-reader.git
cd tactile-reader
pip install -e .
```

Python 3.10+. On Linux make sure system Tk is present if you want the GUI:
`sudo apt install python3-tk`.

## Quick start

```bash
# point any of the tools at a capture directory
tactile-reader-viz       /path/to/052211
tactile-reader-crop-gui  /path/to/052211
tactile-reader-crop      /path/to/052211 --start 24 --end 287
```

Or, from inside a capture directory:

```bash
cd /path/to/052211
tactile-reader-viz
tactile-reader-crop-gui
tactile-reader-crop --start-pct 0.10 --end-pct 0.90
```

## Expected dataset layout

```
<capture_dir>/
  frames.parquet      one row per frame; per-hand pressure (12 finger + 60 palm taxels),
                      per-finger force (N, cal), bend angles, hand IMU, paths to images.
  oak_imu.parquet     high-rate accel + gyro (host-monotonic timestamps)
  rgb/000000.jpg ...  RGB stream
  mono_left/          (optional) left  grayscale stream
  mono_right/         (optional) right grayscale stream
  calib.json          { gloves: {lh,rh}: {bend_min,max, tactile_zero_finger,palm},
                        glove_force: {lh,rh}: {finger: c3,c2,c1,c0 poly} }
```

See `examples/sample_capture_schema.md` for the full column list expected in `frames.parquet`.

## Crop output

`tactile-reader-crop <dir> --start S --end E` produces `<dir>_cropped/`:

```
<dir>_cropped/
  frames.parquet     subset rows, frame_idx renumbered from 0
  oak_imu.parquet    IMU samples mapped into the kept window
  rgb/000000.jpg ... renamed contiguously to match new frame_idx
  mono_left/         (if present in source)
  mono_right/        (if present in source)
  calib.json         copied from source
  crop_meta.json     {source, start_frame, end_frame, start_ts, end_ts, n_frames}
```

The source directory is never modified. If the output directory already exists,
a numbered suffix is appended.

## Range selection options

All three forms work on the same `[start, end]` (inclusive) range; pick whichever
matches what you read off the viewer:

| Bound | By frame_idx       | By percent  | By seconds-from-start |
|-------|--------------------|-------------|-----------------------|
| start | `--start 24`       | `--start-pct 0.1` | `--start-ts 3.5`  |
| end   | `--end 287`        | `--end-pct 0.9`   | `--end-ts 27.0`   |

`--dry-run` prints what would happen without writing.

## License

MIT — see `LICENSE`.
