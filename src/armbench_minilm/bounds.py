"""Model-size accounting bounds derived from an ONNX graph."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto


def _elements(tensor: onnx.TensorProto) -> int:
    return math.prod(tensor.dims)


def _packed_bytes(elements: int, bits: int) -> int:
    return math.ceil(elements * bits / 8)


def analyze_size_bounds(
    baseline_path: Path,
    *,
    candidate_path: Path | None = None,
    target_ops: Sequence[str] = ("MatMul", "Gemm"),
) -> dict[str, Any]:
    """Calculate payload-only precision bounds for constant operator weights."""

    baseline_path = baseline_path.resolve()
    model = onnx.load(baseline_path, load_external_data=False)
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    target_op_set = set(target_ops)
    target_names = {
        node.input[1]
        for node in model.graph.node
        if node.op_type in target_op_set and len(node.input) > 1 and node.input[1] in initializers
    }

    float_initializers = [
        tensor for tensor in initializers.values() if tensor.data_type == TensorProto.FLOAT
    ]
    target_float_initializers = [
        initializers[name]
        for name in target_names
        if initializers[name].data_type == TensorProto.FLOAT
    ]
    all_float_elements = sum(_elements(tensor) for tensor in float_initializers)
    target_float_elements = sum(_elements(tensor) for tensor in target_float_initializers)
    non_target_float_elements = all_float_elements - target_float_elements
    non_target_float_bytes = _packed_bytes(non_target_float_elements, 32)

    result: dict[str, Any] = {
        "schema_version": 1,
        "baseline": {
            "path": baseline_path.name,
            "file_bytes": baseline_path.stat().st_size,
            "float32_initializer_count": len(float_initializers),
            "float32_initializer_elements": all_float_elements,
            "float32_initializer_payload_bytes": _packed_bytes(all_float_elements, 32),
        },
        "target_scope": {
            "operator_types": sorted(target_op_set),
            "float32_initializer_count": len(target_float_initializers),
            "float32_initializer_elements": target_float_elements,
            "float32_initializer_payload_bytes": _packed_bytes(target_float_elements, 32),
            "non_target_float32_payload_bytes": non_target_float_bytes,
            "int8_payload_bound_bytes": non_target_float_bytes
            + _packed_bytes(target_float_elements, 8),
            "int4_payload_bound_bytes": non_target_float_bytes
            + _packed_bytes(target_float_elements, 4),
        },
        "all_float32_initializers": {
            "int8_payload_bound_bytes": _packed_bytes(all_float_elements, 8),
            "int4_payload_bound_bytes": _packed_bytes(all_float_elements, 4),
        },
        "notes": [
            "Bounds cover initializer payload only.",
            (
                "Graph metadata, quantization parameters, alignment, and unsupported tensors "
                "add bytes."
            ),
        ],
    }

    if candidate_path is not None:
        candidate_path = candidate_path.resolve()
        candidate_bytes = candidate_path.stat().st_size
        scope_bound = result["target_scope"]["int8_payload_bound_bytes"]
        result["candidate"] = {
            "path": candidate_path.name,
            "file_bytes": candidate_bytes,
            "bytes_above_target_scope_int8_payload_bound": candidate_bytes - scope_bound,
            "percent_above_target_scope_int8_payload_bound": (
                candidate_bytes / scope_bound - 1.0
            )
            * 100.0,
        }

    return result
