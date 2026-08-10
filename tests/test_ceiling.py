import hashlib
import json
from pathlib import Path

import onnx
import pytest
from onnx import TensorProto, helper

from armbench_minilm.ceiling import analyze_ceiling_runs, write_ceiling_reports


def _write_model(path: Path) -> None:
    graph = helper.make_graph(
        [
            helper.make_node(
                "MatMul", ["input", "weight"], ["dense"], name="/dense/MatMul"
            ),
            helper.make_node(
                "MatMul",
                ["input", "attention_rhs"],
                ["attention"],
                name="/attention/MatMul",
            ),
        ],
        "ceiling-test",
        [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2, 2]),
            helper.make_tensor_value_info("attention_rhs", TensorProto.FLOAT, [1, 2, 2]),
        ],
        [helper.make_tensor_value_info("dense", TensorProto.FLOAT, [1, 2, 3])],
        [helper.make_tensor("weight", TensorProto.FLOAT, [2, 3], [0.0] * 6)],
    )
    onnx.save(helper.make_model(graph), path)


def _event(
    name: str,
    operator: str,
    duration_us: int,
    input_shapes: object,
) -> dict[str, object]:
    return {
        "cat": "Node",
        "name": f"{name}_kernel_time",
        "dur": duration_us,
        "args": {
            "op_name": operator,
            "input_type_shape": input_shapes,
            "output_type_shape": {"float": [1, 2, 3]},
        },
    }


def _write_profile(path: Path, *, optimized: bool) -> None:
    events: list[dict[str, object]] = []
    for _ in range(4):
        events.append(
            _event(
                "/dense/MatMul_quant" if optimized else "/dense/MatMul",
                "DynamicQuantizeMatMul" if optimized else "MatMul",
                15 if optimized else 30,
                {"float": [1, 2, 2]},
            )
        )
        events.append(
            _event(
                "/attention/MatMul",
                "MatMul",
                10,
                [{"float": [1, 2, 2]}, {"float": [1, 2, 2]}],
            )
        )
        events.append(_event("/Add", "Add", 10, {"float": [1, 2, 3]}))
    path.write_text(json.dumps(events), encoding="utf-8")


def test_analyze_ceiling_matches_only_constant_weight_nodes(tmp_path: Path) -> None:
    model_path = tmp_path / "baseline.onnx"
    evidence_dir = tmp_path / "evidence"
    profile_dir = evidence_dir / "profiles"
    profile_dir.mkdir(parents=True)
    _write_model(model_path)
    _write_profile(profile_dir / "baseline.json", optimized=False)
    _write_profile(profile_dir / "optimized.json", optimized=True)
    benchmark = {
        "models": {
            "baseline": {"sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}
        },
        "configuration": {"intra_op_threads": 4},
        "machine": {
            "github_run_id": "1",
            "github_sha": "def",
            "github_runner_image_version": "test",
            "linux_cpu": {"identity": {"cpu_part": "test"}},
        },
        "batches": [
            {"batch_size": 1, "sequence_length": 2, "median_latency_speedup": 1.5}
        ],
        "profiles": [
            {
                "batch_size": 1,
                "sequence_length": 2,
                "baseline": {"profile_file": "baseline.json"},
                "optimized": {"profile_file": "optimized.json"},
            }
        ],
    }
    (evidence_dir / "benchmark.json").write_text(json.dumps(benchmark), encoding="utf-8")

    memory_paths = []
    for index, bandwidth in enumerate((10.0, 30.0), start=1):
        path = tmp_path / f"memory-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "machine": {"github_run_id": str(index)},
                    "results": [
                        {
                            "threads": 4,
                            "size_bytes": 1024,
                            "median_gbps": bandwidth,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        memory_paths.append(path)

    result = analyze_ceiling_runs(
        model_path,
        [evidence_dir],
        memory_microbenchmark_paths=memory_paths,
    )
    case = result["runs"][0]["cases"][0]

    assert result["target_scope"]["constant_weight_nodes"] == 1
    assert result["runs"][0]["evidence_id"] == "1"
    assert "evidence_dir" not in result["runs"][0]
    assert case["baseline_target_node_time_share"] == pytest.approx(0.6)
    assert case["operator_scope_infinite_speedup_limit"] == pytest.approx(2.5)
    assert case["profile_target_pipeline_speedup"] == pytest.approx(2.0)
    assert case["amdahl_speedup_at_profiled_target_rate"] == pytest.approx(1 / 0.7)
    assert case["target_fp32_flops"] == 24
    assert case["dynamic_attention_matmul_flops"] == 16
    assert case["target_fp32_logical_bytes"] == 64
    assert case["target_fp32_arithmetic_intensity_flops_per_logical_byte"] == 0.375
    assert case["memory_projection"]["median_gbps"] == 20.0
    assert case["memory_projection"]["run_median_cv_percent"] == 50.0

    reports = write_ceiling_reports(result, tmp_path / "reports")
    assert reports["json"].is_file()
    assert "2.50x" in reports["markdown"].read_text(encoding="utf-8")
