from __future__ import annotations

import cv2
import numpy as np


RAW_LEN = 256

LEFT_FOOT_MAP = [
    224, 208, 192, 176, 160, 144,
    223, 207, 191, 175, 159, 143,
    222, 206, 190, 174, 158, 142,
    221, 205, 189, 173, 157, 141,
    220, 204, 188, 172, 156, 140,
    219, 203, 187, 171, 155, 139,
    218, 202, 186, 170, 154, 138,
    217, 201, 185, 169, 153, 137,
    216, 200, 184, 168, 152, 136,
    215, 199, 183, 167, 151, 135,
]

RIGHT_FOOT_MAP = [
    23, 7, 247, 231, 215, 199,
    24, 8, 248, 232, 216, 200,
    25, 9, 249, 233, 217, 201,
    26, 10, 250, 234, 218, 202,
    27, 11, 251, 235, 219, 203,
    28, 12, 252, 236, 220, 204,
    29, 13, 253, 237, 221, 205,
    30, 14, 254, 238, 222, 206,
    31, 15, 255, 239, 223, 207,
    32, 16, 256, 240, 224, 208,
]

WB_FRONT_CHEST_MAP = [
    195, 211, 227, 243, 3, 19, 35, 51,
    196, 212, 228, 244, 4, 20, 36, 52,
    197, 213, 229, 245, 5, 21, 37, 53,
    198, 214, 230, 246, 6, 22, 38, 54,
    199, 215, 231, 247, 7, 23, 39, 55,
    200, 216, 232, 248, 8, 24, 40, 56,
]

WB_BACK_MAP = [
    58, 42, 26, 10, 250, 234, 218, 202,
    59, 43, 27, 11, 251, 235, 219, 203,
    60, 44, 28, 12, 252, 236, 220, 204,
    61, 45, 29, 13, 253, 237, 221, 205,
    62, 46, 30, 14, 254, 238, 222, 206,
]

WB_LEFT_ARM_MAP = [
    79, 95, 111, 127,
    80, 96, 112, 128,
]

WB_RIGHT_ARM_MAP = [
    177, 162, 146, 130,
    178, 161, 145, 129,
]

WB_LEFT_SHOULDER_MAP = [249, 233, 217, 201]
WB_RIGHT_SHOULDER_MAP = [9, 25, 41, 57]


def _raw(raw_values: list[int] | bytes | bytearray) -> np.ndarray:
    if isinstance(raw_values, (bytes, bytearray)):
        raw = np.frombuffer(raw_values[:RAW_LEN], dtype=np.uint8)
    else:
        raw = np.asarray(raw_values[:RAW_LEN], dtype=np.uint8)
    if raw.size < RAW_LEN:
        padded = np.zeros(RAW_LEN, dtype=np.uint8)
        padded[: raw.size] = raw
        return padded
    return raw


def _v(raw: np.ndarray, one_indexed: int) -> int:
    if one_indexed <= 0 or one_indexed > raw.size:
        return 0
    return int(raw[one_indexed - 1])


def _map(raw: np.ndarray, mapping: list[int]) -> np.ndarray:
    return np.array([_v(raw, i) for i in mapping], dtype=np.uint8)


def _colorize(arr: np.ndarray, scale: int = 16) -> np.ndarray:
    img = np.asarray(arr, dtype=np.uint8)
    img = np.clip(img.astype(np.float32) * 3.0, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    if scale > 1:
        color = cv2.resize(
            color,
            (img.shape[1] * scale, img.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )
    return color


def render_sensor_panels(
    sensor: str,
    raw_values: list[int] | bytes | bytearray,
    baseline_values: list[int] | bytes | bytearray | None = None,
    deadzone: int = 12,
) -> dict[str, np.ndarray]:
    raw = _raw(raw_values)
    if baseline_values is not None:
        baseline = _raw(baseline_values)
        raw = np.clip(
            raw.astype(np.int16) - baseline.astype(np.int16) - int(deadzone), 0, 255
        ).astype(np.uint8)
    if sensor == "LF":
        flat = _map(raw, LEFT_FOOT_MAP)
        taxels = np.flipud(flat.reshape(10, 6).T)
        return {"LF": _colorize(taxels)}
    if sensor == "RF":
        flat = _map(raw, RIGHT_FOOT_MAP)
        taxels = np.flipud(flat.reshape(10, 6).T)
        return {"RF": _colorize(taxels)}
    if sensor == "WB":
        return {
            "WB_front": _colorize(_map(raw, WB_FRONT_CHEST_MAP).reshape(6, 8)),
            "WB_back": _colorize(_map(raw, WB_BACK_MAP).reshape(5, 8)),
            "WB_left_shoulder": _colorize(_map(raw, WB_RIGHT_SHOULDER_MAP).reshape(1, 4), scale=24),
            "WB_right_shoulder": _colorize(_map(raw, WB_LEFT_SHOULDER_MAP).reshape(1, 4), scale=24),
            "WB_left_arm": _colorize(np.rot90(_map(raw, WB_LEFT_ARM_MAP).reshape(2, 4), k=-1), scale=24),
            "WB_right_arm": _colorize(np.rot90(_map(raw, WB_RIGHT_ARM_MAP).reshape(2, 4), k=-1), scale=24),
        }
    return {}
