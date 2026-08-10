from pathlib import Path

import onnx
from onnx import TensorProto, helper

from armbench_minilm.bounds import analyze_size_bounds


def _write_test_model(path: Path) -> None:
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["input", "weight"], ["projected"]),
            helper.make_node("Add", ["projected", "bias"], ["output"]),
        ],
        "size-bound-test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])],
        [
            helper.make_tensor("weight", TensorProto.FLOAT, [2, 3], [0.0] * 6),
            helper.make_tensor("bias", TensorProto.FLOAT, [3], [0.0] * 3),
        ],
    )
    model = helper.make_model(graph)
    onnx.save(model, path)


def test_analyze_size_bounds_counts_target_and_non_target_initializers(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.onnx"
    _write_test_model(baseline)

    result = analyze_size_bounds(baseline)

    assert result["baseline"]["float32_initializer_elements"] == 9
    assert result["baseline"]["float32_initializer_payload_bytes"] == 36
    assert result["target_scope"]["float32_initializer_elements"] == 6
    assert result["target_scope"]["non_target_float32_payload_bytes"] == 12
    assert result["target_scope"]["int8_payload_bound_bytes"] == 18
    assert result["target_scope"]["int4_payload_bound_bytes"] == 15
    assert result["all_float32_initializers"]["int8_payload_bound_bytes"] == 9
    assert result["all_float32_initializers"]["int4_payload_bound_bytes"] == 5


def test_analyze_size_bounds_reports_candidate_gap(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.onnx"
    candidate = tmp_path / "candidate.onnx"
    _write_test_model(baseline)
    candidate.write_bytes(b"x" * 20)

    result = analyze_size_bounds(baseline, candidate_path=candidate)

    assert result["candidate"]["bytes_above_target_scope_int8_payload_bound"] == 2
    assert result["candidate"]["percent_above_target_scope_int8_payload_bound"] == (
        20 / 18 - 1
    ) * 100
