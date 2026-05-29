# Expected `frames.parquet` schema

The viewer and crop tool assume the columns below in `<capture_dir>/frames.parquet`.
Image paths are relative to the capture directory.

| Column                            | Type                  | Notes                                    |
|-----------------------------------|-----------------------|------------------------------------------|
| `frame_idx`                       | int                   | 0..N-1, monotonic                        |
| `timestamp`                       | float (Unix seconds)  | source clock for the rerun timeline      |
| `rgb_path`                        | str                   | e.g. `"rgb/000000.jpg"`                  |
| `mono_left_path`, `mono_right_path` | str                 | optional; viewer skips if missing        |
| `oak_imu_quaternion`              | float[4] or None      |                                          |
| `oak_imu_gyro`, `oak_imu_accel`   | float[3] or None      |                                          |
| **Per hand `{side}` ∈ {lh, rh}**  |                       |                                          |
| `{side}_glove_timestamp`          | int / float           | sensor's own clock                       |
| `{side}_glove_dt_ms`              | float                 |                                          |
| `{side}_thumb_pressure`           | uint8[12]             | 12-taxel finger pressure, raw            |
| `{side}_index_pressure`           | uint8[12]             |                                          |
| `{side}_middle_pressure`          | uint8[12]             |                                          |
| `{side}_ring_pressure`            | uint8[12]             |                                          |
| `{side}_little_pressure`          | uint8[12]             |                                          |
| `{side}_palm_pressure`            | uint8[60]             | 60-taxel palm pressure                   |
| `{side}_finger_bend`              | uint8[5]              | per-finger raw bend angle (deg)          |
| `{side}_imu_quaternion`           | float[4]              |                                          |
| `{side}_imu_gyro`, `{side}_imu_accel` | float[3]          |                                          |
| `{side}_{finger}_force_N`         | float[12]             | per-taxel force, raw                     |
| `{side}_{finger}_force_N_cal`     | float[12]             | per-taxel force, **calibrated**          |
| `{side}_{finger}_pressure_cal`    | uint8[12]             | finger pressure with tactile_zero removed |
| `{side}_palm_pressure_cal`        | uint8[60]             | palm pressure with tactile_zero removed  |
| `{side}_finger_bend_cal`          | float[5]              | bend normalized to [0, 1] via calib      |

`{finger}` ranges over `thumb, index, middle, ring, little`.

The viewer uses only the `*_cal` columns. The crop tool preserves everything.

## `calib.json`

```jsonc
{
  "camera": { ... },
  "gloves": {
    "lh": {
      "bend_min": { "thumb": 0.0, "index": 1.28, ... },
      "bend_max": { "thumb": 37.0, "index": 39.9, ... },
      "tactile_zero_finger": { "thumb": [12 floats], "index": [...], ... },
      "tactile_zero_palm":   [60 floats]
    },
    "rh": { ... same shape }
  },
  "glove_force": {
    "lh": {
      "thumb":  { "c3": ..., "c2": ..., "c1": ..., "c0": ... },
      "index":  { ... }, ...
    },
    "rh": { ... }
  }
}
```

The force polynomial is `force_N = max(0, c3*p^3 + c2*p^2 + c1*p + c0)` where `p`
is the per-taxel raw pressure value.

## `oak_imu.parquet`

| Column        | Type                              |
|---------------|-----------------------------------|
| `timestamp`   | float (host monotonic, **not** Unix) |
| `accel_x/y/z` | float                             |
| `gyro_x/y/z`  | float                             |

The viewer affine-remaps this clock onto the frames timeline so they share one
scrubbable timeline in the rerun viewer.
