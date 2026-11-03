"""Tactile snapshot -> fixed-width arrays for logging.

Every frame is 256 taxel bytes + 16 IMU bytes per board (see
``jq_tactile_driver``), so a missing or short frame is zero-padded rather than
changing shape mid-episode.
"""

from __future__ import annotations

import numpy as np

RAW_LEN = 256
IMU_LEN = 16


def sensor_arrays(snapshot: dict, sensor: str) -> tuple[np.ndarray, np.ndarray]:
    """``(raw_u8[256], imu_u8[16])`` for one sensor, zero-padded if absent."""
    raw = np.zeros(RAW_LEN, np.uint8)
    imu = np.zeros(IMU_LEN, np.uint8)
    s = snapshot.get(sensor)
    if s:
        r = np.frombuffer(s["raw_sensor_bytes"], np.uint8)
        i = np.frombuffer(s["raw_imu_bytes"], np.uint8)
        raw[: min(RAW_LEN, r.size)] = r[:RAW_LEN]
        imu[: min(IMU_LEN, i.size)] = i[:IMU_LEN]
    return raw, imu


def sample_fields(snapshot: dict, sensors) -> dict:
    """``{tactile_<sensor>_raw_u8, tactile_<sensor>_imu_u8}`` for each sensor."""
    out = {}
    for sensor in sensors:
        raw, imu = sensor_arrays(snapshot, sensor)
        tag = sensor.lower()
        out[f"tactile_{tag}_raw_u8"] = raw
        out[f"tactile_{tag}_imu_u8"] = imu
    return out
