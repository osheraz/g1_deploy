"""Fail-closed inference requirements for robot-facing SONIC launchers."""

from __future__ import annotations

from pathlib import Path

from sonic.model.inference_backend import DIRECT_TENSORRT_PROVIDER

_TRT_NAMES = {"trt", "tensorrt", "tensor_rt"}
_PLANNER_MODES = {"pico", "planner", "smpl"}


def _backend_name(value) -> str:
    return str(value).strip().lower().replace("-", "_")


def require_hardware_config(config, mode: str) -> None:
    """Reject fallback or reduced-precision math before hardware setup."""
    if mode == "record" or getattr(config, "policy_type", None) != "sonic":
        return
    if _backend_name(config.sonic_inference_backend) not in _TRT_NAMES:
        raise RuntimeError(
            "hardware SONIC requires sonic_inference_backend=tensorrt; "
            "automatic or ONNX Runtime fallback is forbidden"
        )
    if mode in _PLANNER_MODES and _backend_name(config.sonic_planner_backend) not in _TRT_NAMES:
        raise RuntimeError(
            "hardware SONIC planner modes require sonic_planner_backend=tensorrt; "
            "automatic or ONNX Runtime fallback is forbidden"
        )
    if bool(config.sonic_tensorrt_fp16):
        raise RuntimeError(
            "hardware SONIC requires the reviewed FP32 model contract; "
            "FP16 needs a separate numerical validation"
        )
    if bool(config.sonic_tensorrt_tf32):
        raise RuntimeError("hardware SONIC forbids TF32 because planner parity exceeds tolerance")


def _require_direct_session(label: str, providers, engine_path) -> Path:
    providers = tuple(providers)
    if providers != (DIRECT_TENSORRT_PROVIDER,):
        raise RuntimeError(
            f"hardware {label} provider is {providers}; expected only "
            f"{DIRECT_TENSORRT_PROVIDER!r}"
        )
    if engine_path is None:
        raise RuntimeError(f"hardware {label} did not report a TensorRT engine path")
    path = Path(engine_path).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"hardware {label} TensorRT engine is missing: {path}")
    return path


def verify_hardware_sessions(policy_runner, mode: str) -> dict[str, Path]:
    """Verify actual providers and cache files before DDS can be initialized."""
    if policy_runner is None:
        return {}
    engines = {
        "encoder": _require_direct_session(
            "encoder",
            policy_runner.encoder_providers,
            policy_runner.encoder_engine_path,
        ),
        "decoder": _require_direct_session(
            "decoder",
            policy_runner.decoder_providers,
            policy_runner.decoder_engine_path,
        ),
    }
    if mode in _PLANNER_MODES:
        planner = getattr(policy_runner.reference, "planner", None)
        if planner is None:
            raise RuntimeError(f"hardware mode {mode!r} requires a planner session")
        engines["planner"] = _require_direct_session(
            "planner", planner.providers, planner.engine_path
        )
    print("SONIC hardware inference contract: direct TensorRT, FP32, TF32 disabled")
    for name, path in engines.items():
        print(f"  {name}: {path}")
    return engines
