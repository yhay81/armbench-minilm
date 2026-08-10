"""Download the pinned FP32 model and create the INT8 derivative."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import copy2

from huggingface_hub import hf_hub_download
from onnxruntime.quantization import QuantType, quantize_dynamic

from armbench_minilm.constants import (
    BASELINE_NAME,
    MODEL_FILENAME,
    MODEL_ID,
    MODEL_REVISION,
    OPTIMIZED_NAME,
)


@dataclass(frozen=True)
class ModelPaths:
    """Prepared model paths used by the benchmark."""

    baseline: Path
    optimized: Path


def prepare_models(work_dir: Path, *, force: bool = False) -> ModelPaths:
    """Download the revision-pinned ONNX model and quantize MatMul/Gemm weights."""

    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = work_dir / "hf-cache"

    downloaded = Path(
        hf_hub_download(
            repo_id=MODEL_ID,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
            cache_dir=cache_dir,
        )
    )
    baseline = work_dir / BASELINE_NAME
    optimized = work_dir / OPTIMIZED_NAME

    if force or not baseline.exists() or baseline.stat().st_size != downloaded.stat().st_size:
        copy2(downloaded, baseline)

    if force or not optimized.exists():
        quantize_dynamic(
            model_input=str(baseline),
            model_output=str(optimized),
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
            extra_options={
                "WeightSymmetric": True,
                "MatMulConstBOnly": True,
            },
        )

    return ModelPaths(baseline=baseline, optimized=optimized)


def existing_models(work_dir: Path) -> ModelPaths:
    """Return prepared model paths or fail with an actionable message."""

    paths = ModelPaths(
        baseline=work_dir.resolve() / BASELINE_NAME,
        optimized=work_dir.resolve() / OPTIMIZED_NAME,
    )
    missing = [str(path) for path in (paths.baseline, paths.optimized) if not path.is_file()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"missing prepared model(s): {joined}; run 'prepare' first")
    return paths
