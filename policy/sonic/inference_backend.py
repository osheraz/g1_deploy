"""Small inference abstraction for SONIC ONNX and native TensorRT engines.

The hardware path uses TensorRT directly, matching the official C++ deploy.
ONNX Runtime remains available for CPU/offline debugging and golden-trace
comparison.  Both backends expose the subset of ``InferenceSession`` used by
the SONIC tracker and kinematic planner.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple
    dtype: object


class OrtSession:
    def __init__(self, model_path, *, providers=None, **_):
        import onnxruntime as ort

        try:
            ort.preload_dlls()
        except Exception:
            pass
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        available = set(ort.get_available_providers())
        requested = providers or ("CUDAExecutionProvider", "CPUExecutionProvider")
        selected = [p for p in requested if p in available]
        if not selected:
            raise RuntimeError(f"ONNX Runtime has no requested provider; available={sorted(available)}")
        self._session = ort.InferenceSession(
            str(Path(model_path).expanduser()), sess_options=opts, providers=selected
        )

    def get_inputs(self):
        return self._session.get_inputs()

    def get_outputs(self):
        return self._session.get_outputs()

    def get_providers(self):
        return self._session.get_providers()

    def run(self, output_names, feed):
        feed = {name: np.ascontiguousarray(value) for name, value in feed.items()}
        return self._session.run(output_names, feed)


class TensorRTSession:
    """TensorRT 10 explicit-I/O engine backed by PyTorch CUDA allocations."""

    def __init__(
        self,
        model_path,
        *,
        input_shapes: dict[str, tuple[int, ...]] | None = None,
        cache_dir=None,
        fp16: bool = False,
        workspace_gib: float = 2.0,
        **_,
    ):
        import tensorrt as trt
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT backend requires a CUDA-capable PyTorch installation")
        major = int(str(trt.__version__).split(".", 1)[0])
        if major < 10:
            raise RuntimeError("direct SONIC TensorRT backend requires TensorRT 10+")

        self._trt = trt
        self._torch = torch
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.input_shapes = {k: tuple(map(int, v)) for k, v in (input_shapes or {}).items()}
        cache_root = Path(cache_dir or "~/.cache/e_skin/sonic").expanduser()
        cache_root.mkdir(parents=True, exist_ok=True)
        signature = {
            "path": str(self.model_path),
            "size": self.model_path.stat().st_size,
            "mtime_ns": self.model_path.stat().st_mtime_ns,
            "trt": trt.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_capability": torch.cuda.get_device_capability(),
            "fp16": bool(fp16),
            "input_shapes": self.input_shapes,
        }
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:20]
        self.engine_path = cache_root / f"{self.model_path.stem}-{key}.engine"

        # TensorRT requires the logger/runtime to outlive objects created from
        # them; retain both for the full session lifetime.
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        engine_bytes = self.engine_path.read_bytes() if self.engine_path.exists() else None
        if engine_bytes is None:
            engine_bytes = self._build_engine(
                self._logger, fp16=bool(fp16), workspace_gib=float(workspace_gib)
            )
            self.engine_path.write_bytes(engine_bytes)
        self._engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None and self.engine_path.exists():
            warnings.warn(
                f"cached TensorRT engine {self.engine_path} is incompatible; rebuilding"
            )
            engine_bytes = self._build_engine(
                self._logger, fp16=bool(fp16), workspace_gib=float(workspace_gib)
            )
            self.engine_path.write_bytes(engine_bytes)
            self._engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError(f"TensorRT could not deserialize {self.engine_path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("TensorRT could not create an execution context")

        self._inputs, self._outputs = [], []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            spec = TensorSpec(
                name=name,
                shape=tuple(self._engine.get_tensor_shape(name)),
                dtype=trt.nptype(self._engine.get_tensor_dtype(name)),
            )
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self._inputs.append(spec)
            else:
                self._outputs.append(spec)

    def _build_engine(self, logger, *, fp16: bool, workspace_gib: float) -> bytes:
        trt = self._trt
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(self.model_path.read_bytes()):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"TensorRT ONNX parse failed for {self.model_path}:\n{errors}")
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, int(workspace_gib * (1 << 30))
        )
        if fp16:
            if not builder.platform_has_fast_fp16:
                raise RuntimeError("FP16 was requested but this TensorRT platform lacks fast FP16")
            config.set_flag(trt.BuilderFlag.FP16)

        profile = None
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            declared = tuple(tensor.shape)
            if any(int(d) < 0 for d in declared):
                try:
                    shape = self.input_shapes[tensor.name]
                except KeyError as exc:
                    raise ValueError(
                        f"dynamic TensorRT input {tensor.name!r} requires an input_shapes entry"
                    ) from exc
                if profile is None:
                    profile = builder.create_optimization_profile()
                profile.set_shape(tensor.name, shape, shape, shape)
        if profile is not None:
            config.add_optimization_profile(profile)
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(f"TensorRT engine build failed for {self.model_path}")
        return bytes(serialized)

    def get_inputs(self):
        return list(self._inputs)

    def get_outputs(self):
        return list(self._outputs)

    def get_providers(self):
        return ["TensorRTExecutionProvider(direct)"]

    def run(self, output_names, feed):
        torch = self._torch
        missing = [s.name for s in self._inputs if s.name not in feed]
        if missing:
            raise ValueError(f"missing TensorRT inputs: {missing}")
        tensors = {}
        stream = torch.cuda.current_stream()
        for spec in self._inputs:
            array = np.ascontiguousarray(feed[spec.name], dtype=spec.dtype)
            if not self._context.set_input_shape(spec.name, tuple(array.shape)):
                raise ValueError(f"TensorRT rejected shape {array.shape} for {spec.name}")
            tensor = torch.as_tensor(array, device="cuda")
            tensors[spec.name] = tensor
            self._context.set_tensor_address(spec.name, int(tensor.data_ptr()))

        wanted = output_names or [s.name for s in self._outputs]
        output_specs = {s.name: s for s in self._outputs}
        for name in wanted:
            if name not in output_specs:
                raise ValueError(f"unknown TensorRT output {name!r}")
            shape = tuple(self._context.get_tensor_shape(name))
            if any(int(d) < 0 for d in shape):
                raise RuntimeError(f"unresolved TensorRT output shape {shape} for {name}")
            np_dtype = np.dtype(output_specs[name].dtype)
            torch_dtype = torch.from_numpy(np.empty((), dtype=np_dtype)).dtype
            tensor = torch.empty(shape, dtype=torch_dtype, device="cuda")
            tensors[name] = tensor
            self._context.set_tensor_address(name, int(tensor.data_ptr()))

        if not self._context.execute_async_v3(stream_handle=int(stream.cuda_stream)):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        stream.synchronize()
        return [tensors[name].cpu().numpy() for name in wanted]


def create_session(
    model_path,
    *,
    backend: str = "auto",
    input_shapes: dict[str, tuple[int, ...]] | None = None,
    cache_dir=None,
    fp16: bool = False,
):
    """Create ``TensorRTSession`` or ``OrtSession`` with explicit fallback rules."""
    choice = str(backend).strip().lower().replace("-", "_")
    if choice in {"trt", "tensorrt", "tensor_rt"}:
        return TensorRTSession(
            model_path, input_shapes=input_shapes, cache_dir=cache_dir, fp16=fp16
        )
    if choice in {"ort", "onnxruntime", "onnx_runtime"}:
        return OrtSession(model_path)
    if choice != "auto":
        raise ValueError("inference backend must be auto, tensorrt, or onnxruntime")
    try:
        return TensorRTSession(
            model_path, input_shapes=input_shapes, cache_dir=cache_dir, fp16=fp16
        )
    except (ImportError, RuntimeError, ValueError, FileNotFoundError) as exc:
        warnings.warn(f"TensorRT unavailable; falling back to ONNX Runtime: {exc}")
        return OrtSession(model_path)
