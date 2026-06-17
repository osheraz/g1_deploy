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

LH_FINGER_PRESSURE = {
    "little": [31, 30, 15, 14, 13, 29, 255, 254, 253, 239, 238, 237],
    "ring": [28, 27, 12, 11, 10, 26, 251, 250, 249, 235, 234, 233],
    "middle": [25, 24, 9, 8, 7, 23, 248, 247, 246, 232, 231, 230],
    "index": [22, 21, 6, 5, 4, 20, 245, 244, 243, 229, 228, 227],
    "thumb": [19, 18, 3, 2, 1, 17, 242, 241, 240, 226, 225, 224],
}

LH_FINGER_BEND = {
    "little": 222,
    "ring": 219,
    "middle": 216,
    "index": 213,
    "thumb": 210,
}

LH_PALM_MAP = [
    207, 206, 205, 204, 203, 202, 201, 200, 199, 198, 197, 196,
    191, 190, 189, 188, 187, 186, 185, 184, 183, 182, 181, 180, 179, 178, 177,
    175, 174, 173, 172, 171, 170, 169, 168, 167, 166, 165, 164, 163, 162, 161,
    159, 158, 157, 156, 155, 154, 153, 152, 151, 150, 149, 148, 147, 146, 145,
    143, 142, 141, 140, 139, 138, 137, 136, 135, 134, 133, 132, 131, 130, 129,
]

RH_FINGER_PRESSURE = {
    "thumb": [240, 239, 238, 256, 255, 254, 16, 15, 14, 32, 31, 30],
    "index": [237, 236, 235, 253, 252, 251, 13, 12, 11, 29, 28, 27],
    "middle": [234, 233, 232, 250, 249, 248, 10, 9, 8, 26, 25, 24],
    "ring": [231, 230, 229, 247, 246, 245, 7, 6, 5, 23, 22, 21],
    "little": [228, 227, 226, 244, 243, 242, 4, 3, 2, 20, 19, 18],
}

RH_FINGER_BEND = {
    "thumb": 47,
    "index": 44,
    "middle": 41,
    "ring": 38,
    "little": 35,
}

RH_PALM_MAP = [
    61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50,
    80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66,
    96, 95, 94, 93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82,
    112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98,
    128, 127, 126, 125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114,
]


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


def _stack_vertical(images: list[np.ndarray], gap: int = 4) -> np.ndarray:
    width = max(img.shape[1] for img in images)
    padded = []
    for img in images:
        if img.shape[1] < width:
            pad = np.zeros((img.shape[0], width - img.shape[1], 3), dtype=np.uint8)
            img = np.concatenate([img, pad], axis=1)
        padded.append(img)

    separator = np.zeros((gap, width, 3), dtype=np.uint8)
    out = padded[:1]
    for img in padded[1:]:
        out.extend([separator, img])
    return np.concatenate(out, axis=0)


def _render_hand_panel(
    raw: np.ndarray,
    sensor: str,
    palm_map: list[int],
    finger_pressure: dict[str, list[int]],
    finger_bend: dict[str, int],
) -> np.ndarray:
    palm_flat = _map(raw, palm_map)
    palm_grid = np.zeros(75, dtype=np.uint8)
    palm_grid[: palm_flat.size] = palm_flat
    palm = _colorize(palm_grid.reshape(5, 15), scale=12)

    finger_order = ["thumb", "index", "middle", "ring", "little"]
    fingers = [
        _colorize(_map(raw, finger_pressure[name]).reshape(4, 3), scale=12)
        for name in finger_order
    ]
    finger_panel = np.concatenate(fingers, axis=1)

    bend = np.array(
        [[_v(raw, finger_bend[name]) for name in finger_order]],
        dtype=np.uint8,
    )
    bend_panel = _colorize(bend, scale=36)

    panel = _stack_vertical([palm, finger_panel, bend_panel], gap=4)
    cv2.putText(
        panel,
        sensor,
        (6, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        1,
        cv2.LINE_AA,
    )
    return panel


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
    if sensor == "LH":
        return {
            "LH": _render_hand_panel(
                raw,
                sensor,
                LH_PALM_MAP,
                LH_FINGER_PRESSURE,
                LH_FINGER_BEND,
            )
        }
    if sensor == "RH":
        return {
            "RH": _render_hand_panel(
                raw,
                sensor,
                RH_PALM_MAP,
                RH_FINGER_PRESSURE,
                RH_FINGER_BEND,
            )
        }
    return {}
