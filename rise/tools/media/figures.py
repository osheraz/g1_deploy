"""Optional figure overlays/strips. Loaded only for an explicit export."""

import json
from pathlib import Path
from rise.paths import (
    ROOT,
    RESULTS_ROOT,
    PAPER_RESULTS_ROOT,
    HIGHLIGHTS_ROOT,
    PAPER_HIGHLIGHTS_ROOT,
)

HIGHLIGHTS = HIGHLIGHTS_ROOT
_segmenter = None


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def highlight_directory(session):
    """New-run exports stay in logs; explicitly opened paper archives stay there."""
    for root, highlight in (
        (RESULTS_ROOT, HIGHLIGHTS),
        (PAPER_RESULTS_ROOT, PAPER_HIGHLIGHTS_ROOT),
    ):
        try:
            rel = session.relative_to(root)
        except ValueError:
            continue
        return highlight / rel
    return None


def highlight_picks(highlight):
    """Your picked frames frames/NN_*.png (or NN_*.jpg), in order."""
    return sorted(
        p
        for p in (highlight / "frames").glob("[0-9][0-9]_*")
        if p.suffix.lower() in (".png", ".jpg", ".jpeg")
    )


def people_mask(image, moving=None):
    """People in a frame (spotters, operators), so they are not ghosted.
    Mask R-CNN also calls the humanoid a person; the robot is silver and black,
    so an instance counts as a person only on positive evidence: skin tones, or
    saturated clothing with little white shell (bright, unsaturated pixels). Colour is
    read on the instance's moving pixels when `moving` is given, so a static tan
    background (plywood) leaking into a loose mask does not pass for skin."""
    import cv2, numpy as np, torch

    global _segmenter
    if _segmenter is None:
        from torchvision.models.detection import (
            maskrcnn_resnet50_fpn_v2,
            MaskRCNN_ResNet50_FPN_V2_Weights,
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        _segmenter = (
            maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
            .eval()
            .to(device),
            device,
        )
    model, device = _segmenter
    tensor = (
        torch.from_numpy(np.ascontiguousarray(image[..., ::-1]))
        .permute(2, 0, 1)
        .float()
        .div(255)
        .to(device)
    )
    with torch.no_grad():
        found = model([tensor])[0]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    white = (saturation < 40) & (value > 150)
    skin = (hue <= 20) & (saturation >= 45) & (saturation <= 170) & (value >= 90)
    people = np.zeros(image.shape[:2], bool)
    for score, label, mask in zip(found["scores"], found["labels"], found["masks"]):
        if int(label) != 1 or float(score) < 0.2:
            continue
        region = (mask[0] > 0.5).cpu().numpy()
        if not region.any():
            continue
        sample = (
            region & moving if moving is not None and (region & moving).sum() >= 500 else region
        )
        if skin[sample].mean() >= 0.08 or (
            white[sample].mean() < 0.15 and np.percentile(saturation[sample], 85) >= 90
        ):
            people |= region
    return cv2.dilate(people.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)


def robot_cutouts(frames):
    """Scene (per-pixel median of the frames) and one feathered robot mask per frame:
    where the frame differs from the scene, people excluded, small specks dropped."""
    import cv2, numpy as np

    scene = np.median(np.stack(frames), axis=0).astype(np.uint8)
    scene_blur = cv2.GaussianBlur(scene, (7, 7), 0)
    scene_people = people_mask(scene)  # someone standing still in most picks ends up in the scene
    masks = []
    for img in frames:
        m = (cv2.absdiff(cv2.GaussianBlur(img, (7, 7), 0), scene_blur).max(axis=2) > 28).astype(
            np.uint8
        )
        m[people_mask(img, m.astype(bool)) | scene_people] = 0
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m)
        keep = np.zeros(m.shape, np.uint8)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] > 2500:
                keep[labels == k] = 1
        keep = cv2.dilate(keep, np.ones((5, 5), np.uint8))
        masks.append(cv2.GaussianBlur(keep.astype(np.float32), (9, 9), 0)[..., None])
    return scene, masks


def robot_blob(image, mask, previous=None):
    """Bounding box (x, y, w, h) of the robot in one pick: the largest blob that shows
    the robot's white shell and, given the previous pick's box, lies near it (the robot
    moves gradually between picks; a spotter elsewhere does not); None when the mask is empty."""
    import cv2, numpy as np

    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask[..., 0] > 0.5).astype(np.uint8))
    if n < 2:
        return None
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = (hsv[..., 1] < 40) & (hsv[..., 2] > 150)
    boxes = [tuple(int(v) for v in stats[1 + k, :4]) for k in range(n - 1)]
    shell = [white[labels == 1 + k].mean() for k in range(n - 1)]
    areas = stats[1:, cv2.CC_STAT_AREA]
    candidates = [k for k in range(n - 1) if shell[k] >= 0.08] or list(range(n - 1))
    if previous is not None:
        px, py, pw, ph = previous
        reach = max(500, 2 * max(pw, ph))
        near = [
            k
            for k in candidates
            if np.hypot(
                boxes[k][0] + boxes[k][2] / 2 - px - pw / 2,
                boxes[k][1] + boxes[k][3] / 2 - py - ph / 2,
            )
            <= reach
        ]
        candidates = near or candidates
    best = max(candidates, key=lambda k: areas[k])
    x, y, w, h = boxes[best]
    # An occluded robot splits into stacked blobs (legs below a torso); rejoin them.
    for k in candidates:
        bx, by, bw, bh = boxes[k]
        overlap = min(x + w, bx + bw) - max(x, bx)
        gap = max(by - (y + h), y - (by + bh))
        if k != best and overlap >= 0.4 * min(w, bw) and gap <= 120:
            x0, y0 = min(x, bx), min(y, by)
            x1, y1 = max(x + w, bx + bw), max(y + h, by + bh)
            x, y, w, h = x0, y0, x1 - x0, y1 - y0
    return (x, y, w, h)


def robot_boxes(frames, masks):
    """The robot's box in every pick (None where nothing moved), tracked pick to pick."""
    boxes = []
    previous = None
    for f, m in zip(frames, masks):
        box = robot_blob(f, m, previous)
        boxes.append(box)
        previous = box or previous
    return boxes


def robot_box(frames, masks, pad=60):
    """One compact rectangle around the robot across all picks: the robot blob of
    each pick unioned, plus a margin."""
    height, width = masks[0].shape[:2]
    boxes = [b for b in robot_boxes(frames, masks) if b is not None]
    if not boxes:
        return 0, 0, width, height
    return (
        max(0, min(b[0] for b in boxes) - pad),
        max(0, min(b[1] for b in boxes) - pad),
        min(width, max(b[0] + b[2] for b in boxes) + pad),
        min(height, max(b[1] + b[3] for b in boxes) + pad),
    )


def robot_windows(frames, masks, pad=60):
    """One window per pick, all the same size, each centred on that pick's robot:
    the window follows the robot, so people standing beside its path stay out."""
    import numpy as np

    height, width = masks[0].shape[:2]
    boxes = robot_boxes(frames, masks)
    if all(b is None for b in boxes):
        return [(0, 0, width, height)] * len(masks)
    last = next(b for b in boxes if b is not None)
    boxes = [last := (b if b is not None else last) for b in boxes]
    w = min(width, max(b[2] for b in boxes) + 2 * pad)
    h = min(height, max(b[3] for b in boxes) + 2 * pad)
    windows = []
    for x, y, bw, bh in boxes:
        x0 = int(np.clip(x + bw // 2 - w // 2, 0, width - w))
        y0 = int(np.clip(y + bh // 2 - h // 2, 0, height - h))
        windows.append((x0, y0, x0 + w, y0 + h))
    return windows


def export_overlay(session, recording, review, *, opacity=(0.4, 0.7)):
    """Motion overlays from the highlight frames you picked (real_world_hightlight/<run>/frames/NN_*.png).

    The scene is the per-pixel median of the picks. Each pick's robot is cut out as
    where that pick differs from the scene (people excluded) and pasted onto the
    scene with a feathered edge: earlier picks at rising opacity, the last pick fully
    opaque. overlay_12345 uses all picks; every subset keeping the first and last
    pick is written too (overlay_15, overlay_135, overlay_1245, ...). overlay =
    overlay_12345. PNGs go to the highlight run's overlays/, JPEGs to review/.
    """
    import cv2, itertools, numpy as np, shutil

    highlight = highlight_directory(session)
    if highlight is None:
        return None
    picks = highlight_picks(highlight)
    marker = review / "overlay.json"
    if len(picks) < 2:
        return None
    signature = [[p.name, p.stat().st_size] for p in picks]
    method = f"cutout_evidence_{opacity[0]:g}_{opacity[1]:g}"
    previous = read_json(marker)
    if (
        previous.get("picks") == signature
        and previous.get("method") == method
        and (review / "overlay.jpg").is_file()
    ):
        return str(review / "overlay.jpg")
    overlays = highlight / "overlays"
    overlays.mkdir(exist_ok=True)
    for folder in (review, overlays):
        for stale in (
            list(folder.glob("overlay*.jpg"))
            + list(folder.glob("overlay*.png"))
            + list(folder.glob("keyframes*.jpg"))
        ):
            stale.unlink()
    frames = [cv2.imread(str(p)) for p in picks]
    scene, masks = robot_cutouts(frames)
    floats = [f.astype(np.float32) for f in frames]
    x0, y0, x1, y1 = robot_box(frames, masks)
    middle = range(1, len(picks) - 1)
    written = []
    for count in range(len(middle) + 1):
        for chosen in itertools.combinations(middle, count):
            keys = [0, *chosen, len(picks) - 1]
            name = "overlay_" + "".join(str(k + 1) for k in keys)
            canvas = scene.astype(np.float32)
            opacities = np.linspace(opacity[0], opacity[1], len(keys) - 1)
            for a, k in zip(opacities, keys[:-1]):
                alpha = a * masks[k]
                canvas = (1 - alpha) * canvas + alpha * floats[k]
            alpha = masks[keys[-1]]
            canvas = (1 - alpha) * canvas + alpha * floats[keys[-1]]
            image = canvas.astype(np.uint8)[y0:y1, x0:x1]
            cv2.imwrite(str(review / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(overlays / f"{name}.png"), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            written.append(name)
    for folder, ext in ((review, "jpg"), (overlays, "png")):
        shutil.copy(folder / f"{written[-1]}.{ext}", folder / f"overlay.{ext}")
    marker.write_text(
        json.dumps(
            dict(
                picks=signature,
                method=method,
                overlays=written,
                note="Robot cut-outs from the highlight picks pasted onto the median scene; earlier picks at "
                f"{opacity[0]:g}-{opacity[1]:g} opacity (oldest lowest), the last pick opaque; people excluded. "
                "Change the highlight files or --overlay-opacity and rerun --export-overlay.",
            ),
            indent=2,
        )
        + "\n"
    )
    return str(review / "overlay.jpg")


def export_strip(session, recording, review, *, width=384, gap=8):
    """strip_12345: the highlight picks side by side, left to right, each `width` px
    wide, white gaps; every subset keeping the first and last pick is written too.
    strip = all picks. Each thumbnail is the same-size window centred on that pick's
    robot. PNGs go to the highlight run's strips/, JPEGs to review/."""
    import cv2, itertools, numpy as np, shutil

    highlight = highlight_directory(session)
    if highlight is None:
        return None
    picks = highlight_picks(highlight)
    if len(picks) < 2:
        return None
    signature = [[p.name, p.stat().st_size] for p in picks]
    marker = review / "strip.json"
    method = f"strip_evidence_{width}_{gap}"
    previous = read_json(marker)
    if (
        previous.get("picks") == signature
        and previous.get("method") == method
        and (review / "strip.jpg").is_file()
    ):
        return str(review / "strip.jpg")
    strips = highlight / "strips"
    strips.mkdir(exist_ok=True)
    for folder in (review, strips):
        for stale in list(folder.glob("strip*.jpg")) + list(folder.glob("strip*.png")):
            stale.unlink()
    frames = [cv2.imread(str(p)) for p in picks]
    _, masks = robot_cutouts(frames)
    windows = robot_windows(frames, masks)
    thumbs = []
    for img, (x0, y0, x1, y1) in zip(frames, windows):
        crop = img[y0:y1, x0:x1]
        thumbs.append(
            cv2.resize(
                crop,
                (width, round(crop.shape[0] * width / crop.shape[1])),
                interpolation=cv2.INTER_AREA,
            )
        )
    spacer = np.full((thumbs[0].shape[0], gap, 3), 255, np.uint8)
    middle = range(1, len(picks) - 1)
    written = []
    for count in range(len(middle) + 1):
        for chosen in itertools.combinations(middle, count):
            keys = [0, *chosen, len(picks) - 1]
            name = "strip_" + "".join(str(k + 1) for k in keys)
            parts = []
            for k in keys:
                parts += [thumbs[k], spacer]
            image = np.hstack(parts[:-1])
            cv2.imwrite(str(review / f"{name}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(strips / f"{name}.png"), image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            written.append(name)
    for folder, ext in ((review, "jpg"), (strips, "png")):
        shutil.copy(folder / f"{written[-1]}.{ext}", folder / f"strip.{ext}")
    marker.write_text(
        json.dumps(
            dict(
                picks=signature,
                method=method,
                strips=written,
                windows=[[int(v) for v in w] for w in windows],
                note="Highlight picks side by side, left to right, each in a same-size window centred on the robot. "
                "strip uses all picks. Change the picks and rerun --export-strip.",
            ),
            indent=2,
        )
        + "\n"
    )
    return str(review / "strip.jpg")
