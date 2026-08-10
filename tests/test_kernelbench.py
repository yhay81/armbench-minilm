from pathlib import Path

import onnx
from onnx import TensorProto, helper

from armbench_minilm.kernelbench import (
    _target_shape_counts,
    measure_exact_shape_kernels,
    write_kernel_microbenchmark,
)


def _write_model(path: Path) -> None:
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"], name="dense")],
        "kernelbench-test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 3])],
        [helper.make_tensor("weight", TensorProto.FLOAT, [2, 3], [0.1] * 6)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.save(model, path)


def test_exact_shape_kernel_benchmark_retains_balanced_raw_samples(tmp_path: Path) -> None:
    model_path = tmp_path / "baseline.onnx"
    _write_model(model_path)

    assert _target_shape_counts(model_path) == {(2, 3): 1}
    result = measure_exact_shape_kernels(
        model_path,
        rows=[1, 2],
        threads=1,
        warmups=0,
        block_warmups=0,
        iterations=2,
        measurement_blocks=2,
        random_seed=7,
        bootstrap_resamples=10,
    )

    assert result["target_weight_shapes"] == [
        {"k": 2, "n": 3, "target_node_count": 1}
    ]
    assert len(result["results"]) == 2
    for case in result["results"]:
        assert case["flops_per_call"] == 2 * case["rows"] * 2 * 3
        assert len(case["measurement_blocks"]) == 2
        assert all(
            set(block["order"]) == {"fp32", "qint8"}
            for block in case["measurement_blocks"]
        )
        assert sum(
            len(block["samples_ms"][precision])
            for block in case["measurement_blocks"]
            for precision in ("fp32", "qint8")
        ) == 4
        assert case["fp32"]["wall"]["effective_gigaops_per_second"] > 0.0
        assert case["qint8"]["wall"]["effective_gigaops_per_second"] > 0.0

    output = write_kernel_microbenchmark(result, tmp_path / "kernel.json")
    assert output.is_file()
