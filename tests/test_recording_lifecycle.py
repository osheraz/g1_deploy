"""Device-free recording startup, shutdown and child-command checks."""

from pathlib import Path
import subprocess
import time

import numpy as np
import pytest
import zmq

from core.sensing.depth_client import SimpleDepthClient
from rise.recording.session import PolicyRecording, commands
from vigor.recording import RealLogger


def test_depth_receiver_stops_without_a_publisher():
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    port = publisher.bind_to_random_port("tcp://127.0.0.1")
    client = SimpleDepthClient("127.0.0.1", port, 2, 2)
    client.start()
    try:
        time.sleep(0.05)
        client.running = False
        client.join(timeout=1.0)
        assert not client.is_alive()
        assert client.latest_depth is None
    finally:
        client.running = False
        publisher.close(linger=0)
        context.term()


@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("camera", [False, True])
def test_sensor_child_commands_use_executable_current_paths(tmp_path, live, camera):
    recorder = PolicyRecording(
        tmp_path,
        interface="test0",
        seconds=1.0,
        live_odometry=dict(mount_calibration=tmp_path / "mount.json", port=19999) if live else None,
        camera=dict(device="/dev/video-test", size="640x480", fps=30) if camera else None,
    )
    plan = commands(recorder.args, tmp_path)
    assert plan[-1][0] == "capture"
    assert ("pelvis" in dict(plan)) is live
    assert ("camera" in dict(plan)) is camera
    for name, command in plan:
        if command[0] != "/usr/bin/python3":
            continue
        script = Path(command[1])
        assert script.is_file(), (name, script)
        # Parses the real worker; --help exits before ROS/DDS or device initialization.
        result = subprocess.run(
            command[:2] + ["--help"], cwd=tmp_path, text=True, capture_output=True, timeout=20
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_vigor_recording_drains_and_reports_overflow(tmp_path):
    logger = RealLogger(str(tmp_path / "episode.npz"), max_queue=2)
    logger.push({"joint": [1.0, 2.0]})
    logger.push({"joint": [3.0, 4.0]})
    logger.push({"joint": [5.0, 6.0]})
    logger.start()
    logger.stop()
    logger.save()
    with np.load(logger.save_path, allow_pickle=True) as data:
        np.testing.assert_array_equal(data["joint"], [[1.0, 2.0], [3.0, 4.0]])
        assert data["meta"][0]["dropped_samples"] == 1


def test_vigor_missing_weights_fail_before_inference(tmp_path):
    from types import SimpleNamespace
    from vigor.policy_runner import PolicyRunner

    runner = PolicyRunner.__new__(PolicyRunner)
    runner.config = SimpleNamespace(policy_path=tmp_path / "missing.pt")
    with pytest.raises(FileNotFoundError, match="Policy path does not exist"):
        runner._load_policy()


def test_video_and_frame_exports_share_alignment_and_missing_depth(tmp_path, monkeypatch):
    import json
    import cv2
    from rise import replay
    from rise.recording import depth as depth_recording

    recording = tmp_path / "sensors"
    camera = recording / "camera"
    depth_dir = recording / "depth"
    review = tmp_path / "review"
    for directory in (camera, depth_dir, review):
        directory.mkdir(parents=True)
    (depth_dir / "frames.jsonl").touch()
    (depth_dir / "frames.bin").touch()
    (recording / "policy_start.json").write_text(json.dumps({"monotonic_s": 10.0}))
    jpeg = cv2.imencode(".jpg", np.zeros((8, 8, 3), np.uint8))[1].tobytes()
    (camera / "frames.mjpg").write_bytes(jpeg * 4)
    np.savez(
        camera / "frames.npz",
        ts_monotonic_s=[10.0, 11.0, 12.0, 13.0],
        fps=1.0,
        offset=np.arange(4) * len(jpeg),
        bytes=np.full(4, len(jpeg)),
    )
    calls = []

    class Depth:
        view_shape = (8, 8)

        def __init__(self, directory):
            pass

        def indices_at(self, times):
            np.testing.assert_array_equal(times, [10.0, 11.0, 12.0, 13.0])
            return np.array([-1, 0, 0, 1])

        def frame(self, index):
            calls.append(index)
            valid = np.ones((8, 8), bool)
            valid[0, 0] = False
            return np.full((8, 8), 0.5 + index), valid

    monkeypatch.setattr(depth_recording, "RecordedDepth", Depth)
    video = replay.export_depth_video(tmp_path, recording, review)
    frames = Path(replay.export_frames(tmp_path, recording, review))
    assert calls == [
        0,
        1,
        0,
        1,
    ]  # Decode each capture once per export, including repeated timestamps.
    assert (frames / "external/000000.jpg").read_bytes() == jpeg
    missing = cv2.imread(str(frames / "depth/000000.png"))
    assert np.all(missing == (100, 0, 130))
    first = cv2.imread(str(frames / "depth/000001.png"))
    repeated = cv2.imread(str(frames / "depth/000002.png"))
    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(first[0, 0], [100, 0, 130])
    np.testing.assert_array_equal(first[1, 1], [53, 53, 53])
    capture = cv2.VideoCapture(video)
    count = 0
    while capture.read()[0]:
        count += 1
    capture.release()
    assert count == 4
