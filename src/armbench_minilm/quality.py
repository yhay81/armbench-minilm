"""Revision-pinned downstream task-quality gate for Arm64 runtime variants."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from numpy.typing import NDArray
from transformers import AutoTokenizer

from armbench_minilm.benchmark import (
    _create_session,
    _embed,
    _feeds,
    _machine_metadata,
    _model_metadata,
)
from armbench_minilm.constants import (
    BF16_EXPERIMENT_ID,
    MAX_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
)
from armbench_minilm.experiments import bf16_variants
from armbench_minilm.metrics import rowwise_cosine
from armbench_minilm.models import ModelPaths

QUALITY_EXPERIMENT_ID = "r2-bf16-task-quality-v1"

INDIC_STS_REPO_ID = "mteb/IndicCrosslingualSTS"
INDIC_STS_REVISION = "f0366eb5a20087355c0e131162bbed943ba54b51"
INDIC_STS_LICENSE = "CC0-1.0"
INDIC_STS_FILES = {
    "en-as/test-00000-of-00001.parquet": (
        "0f9a84eedd7b7445b189c08efa954a43949eba48a85125b1cce8750c8803949a"
    ),
    "en-bn/test-00000-of-00001.parquet": (
        "e1dad054add6feff8f5a389dcb4378a6dad8c402682024b2873b66b17ad9194a"
    ),
    "en-gu/test-00000-of-00001.parquet": (
        "b16b13f32439ecb7dbba5c725ec67986e648f130c159a5bf7f736b0bf3005ce9"
    ),
    "en-hi/test-00000-of-00001.parquet": (
        "d0da18b0e75f2119ff931180ca2a7a437f22fdfc88ed8989bf024e3a63b9c3b2"
    ),
    "en-kn/test-00000-of-00001.parquet": (
        "0e4e7f4506dafe63863267a9d7469f468a4273a0a014d957cb4169a8b7f23297"
    ),
    "en-ml/test-00000-of-00001.parquet": (
        "132ce945b79a31970fb415072bee8dd0c211ca3e4ff659d654e130e48974ae3a"
    ),
    "en-mr/test-00000-of-00001.parquet": (
        "27700cd8c7dfef7e07db92fec5576858098f9307d3dac13f4278cb5c59342f9f"
    ),
    "en-or/test-00000-of-00001.parquet": (
        "e2f321ab4082803b9af6d51a408dd4779b6881a430ec4913cf5edece2a5318bc"
    ),
    "en-pa/test-00000-of-00001.parquet": (
        "5c7ac31079458317874ceb391b02655a264a5a726b8d45e74b0c5fb18f7060fa"
    ),
    "en-ta/test-00000-of-00001.parquet": (
        "1eae4b4eef9584154d49f14054d5a867d58ce8007f3d3d0439afc79635aeeb45"
    ),
    "en-te/test-00000-of-00001.parquet": (
        "a6e1855e43a059cab997dc651218f3f34a72a159a2819e9aa4e27f6e9c35eb61"
    ),
    "en-ur/test-00000-of-00001.parquet": (
        "6c3373ca999c9b3a54af7c79f9842964a8562531f35855fb1bfe80d331dcfc6b"
    ),
}
INDIC_STS_ROWS_PER_CONFIG = 256

ARGUANA_REPO_ID = "mteb/arguana"
ARGUANA_REVISION = "c22ab2a51041ffd869aaddef7af8d8215647e41a"
ARGUANA_LICENSE = "CC-BY-SA-4.0"
ARGUANA_FILES = {
    "corpus.jsonl": "c5cb4da0f464e93f19577bf4cfb18e9833c3725da12631eb53554628013b0aa9",
    "queries.jsonl": "963c899dd307c9d32416ef47d0701f0be778daeac610af71d735df0d789371a4",
    "qrels/test.jsonl": "391101246d5b5c404222b6ea408fa3924f0b6c6569c99aac028e29c882174b77",
}
ARGUANA_CORPUS_ROWS = 8_674
ARGUANA_QUERY_ROWS = 1_406
ARGUANA_QREL_ROWS = 1_406
ARGUANA_MISSING_QREL_TARGETS = (
    "test-education-ufsdfkhbwu-con03b",
    "test-free-speech-debate-yfsdfkhbwu-con03b",
    "test-politics-dhwem-pro06b",
    "test-science-sghwbdgmo-con03b",
    "test-society-asfhwapg-con04b",
)

QUALITY_COMPARISONS = (
    ("fp32_bf16_vs_control", "fp32_control", "fp32_bf16_fastmath"),
    ("qint8_bf16_vs_control", "qint8_control", "qint8_bf16_fastmath"),
    ("qint8_control_vs_fp32", "fp32_control", "qint8_control"),
    ("qint8_bf16_vs_fp32", "fp32_control", "qint8_bf16_fastmath"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_hash(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Verify one immutable source file and return non-local manifest metadata."""

    actual = _sha256(path)
    if actual != expected_sha256.lower():
        raise ValueError(
            f"source hash mismatch for {path.name}: expected {expected_sha256}, got {actual}"
        )
    return {"sha256": actual, "bytes": path.stat().st_size}


def _download_verified(
    *,
    repo_id: str,
    revision: str,
    files: Mapping[str, str],
    cache_dir: Path,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    paths: dict[str, Path] = {}
    manifest: dict[str, dict[str, Any]] = {}
    for filename, expected_hash in files.items():
        path = Path(
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                revision=revision,
                cache_dir=cache_dir,
            )
        )
        paths[filename] = path
        manifest[filename] = verify_file_hash(path, expected_hash)
    return paths, manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON in {path.name} line {line_number}") from error
            if not isinstance(row, dict):
                raise ValueError(f"expected an object in {path.name} line {line_number}")
            rows.append(row)
    return rows


def load_quality_data(work_dir: Path) -> dict[str, Any]:
    """Download, hash-check, and schema-check the fixed quality-gate data."""

    cache_dir = work_dir.resolve() / "hf-cache"
    sts_paths, sts_manifest = _download_verified(
        repo_id=INDIC_STS_REPO_ID,
        revision=INDIC_STS_REVISION,
        files=INDIC_STS_FILES,
        cache_dir=cache_dir,
    )
    sts_configs: dict[str, dict[str, Any]] = {}
    for filename, path in sts_paths.items():
        config = filename.split("/", 1)[0]
        table = pq.read_table(path, columns=["sentence1", "sentence2", "score"])
        if table.num_rows != INDIC_STS_ROWS_PER_CONFIG:
            raise ValueError(
                f"unexpected {config} rows: {table.num_rows}; "
                f"expected {INDIC_STS_ROWS_PER_CONFIG}"
            )
        payload = table.to_pydict()
        sentence1 = tuple(str(value) for value in payload["sentence1"])
        sentence2 = tuple(str(value) for value in payload["sentence2"])
        scores = np.asarray(payload["score"], dtype=np.float64)
        if len(sentence1) != len(sentence2) or len(sentence1) != scores.size:
            raise ValueError(f"misaligned STS columns in {filename}")
        if not np.isfinite(scores).all():
            raise ValueError(f"non-finite STS score in {filename}")
        sts_configs[config] = {
            "sentence1": sentence1,
            "sentence2": sentence2,
            "scores": scores,
        }
    if set(sts_configs) != {name.split("/", 1)[0] for name in INDIC_STS_FILES}:
        raise ValueError("STS configuration set differs from the fixed manifest")

    arguana_paths, arguana_manifest = _download_verified(
        repo_id=ARGUANA_REPO_ID,
        revision=ARGUANA_REVISION,
        files=ARGUANA_FILES,
        cache_dir=cache_dir,
    )
    corpus_rows = _read_jsonl(arguana_paths["corpus.jsonl"])
    query_rows = _read_jsonl(arguana_paths["queries.jsonl"])
    qrel_rows = _read_jsonl(arguana_paths["qrels/test.jsonl"])
    expected_counts = (ARGUANA_CORPUS_ROWS, ARGUANA_QUERY_ROWS, ARGUANA_QREL_ROWS)
    actual_counts = (len(corpus_rows), len(query_rows), len(qrel_rows))
    if actual_counts != expected_counts:
        raise ValueError(
            f"unexpected ArguAna cardinalities: {actual_counts}; expected {expected_counts}"
        )

    corpus_ids: list[str] = []
    corpus_texts: list[str] = []
    for row in corpus_rows:
        identifier = str(row["_id"])
        title = str(row.get("title") or "")
        body = str(row["text"])
        corpus_ids.append(identifier)
        corpus_texts.append(f"{title} {body}".strip() if title else body.strip())
    query_ids = [str(row["_id"]) for row in query_rows]
    query_texts = [str(row["text"]) for row in query_rows]
    if len(set(corpus_ids)) != len(corpus_ids) or len(set(query_ids)) != len(query_ids):
        raise ValueError("ArguAna contains duplicate corpus or query identifiers")

    qrels: dict[str, dict[str, float]] = {}
    for row in qrel_rows:
        query_id = str(row["query-id"])
        corpus_id = str(row["corpus-id"])
        relevance = float(row["score"])
        if query_id in qrels:
            raise ValueError(f"ArguAna query {query_id} has more than one relevance row")
        qrels[query_id] = {corpus_id: relevance}
    if set(qrels) != set(query_ids):
        raise ValueError("ArguAna qrels do not cover the exact query identifier set")
    corpus_id_set = set(corpus_ids)
    missing_targets = tuple(
        sorted(
            target
            for items in qrels.values()
            for target in items
            if target not in corpus_id_set
        )
    )
    if missing_targets != ARGUANA_MISSING_QREL_TARGETS:
        raise ValueError(
            "ArguAna missing-target set differs from the fixed official-task limitation: "
            f"{missing_targets}"
        )

    return {
        "sts": {
            "configs": sts_configs,
            "manifest": {
                "repo_id": INDIC_STS_REPO_ID,
                "revision": INDIC_STS_REVISION,
                "license": INDIC_STS_LICENSE,
                "files": sts_manifest,
            },
        },
        "retrieval": {
            "corpus_ids": tuple(corpus_ids),
            "corpus_texts": tuple(corpus_texts),
            "query_ids": tuple(query_ids),
            "query_texts": tuple(query_texts),
            "qrels": qrels,
            "missing_qrel_targets": missing_targets,
            "manifest": {
                "repo_id": ARGUANA_REPO_ID,
                "revision": ARGUANA_REVISION,
                "license": ARGUANA_LICENSE,
                "files": arguana_manifest,
            },
        },
    }


def average_ranks(
    values: Sequence[float] | NDArray[np.floating],
) -> NDArray[np.float64]:
    """Return one-based average ranks with exact-tie handling like scipy.rankdata."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("rank input must be a one-dimensional array with at least two values")
    if not np.isfinite(array).all():
        raise ValueError("rank input must be finite")
    order = np.argsort(array, kind="stable")
    ranks = np.empty(array.size, dtype=np.float64)
    start = 0
    while start < array.size:
        end = start + 1
        while end < array.size and array[order[end]] == array[order[start]]:
            end += 1
        average = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average
        start = end
    return ranks


def spearman_correlation(
    left: Sequence[float] | NDArray[np.floating],
    right: Sequence[float] | NDArray[np.floating],
) -> float:
    """Calculate Spearman correlation through average ranks without scipy."""

    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    left_centered = left_ranks - np.mean(left_ranks)
    right_centered = right_ranks - np.mean(right_ranks)
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator == 0.0:
        raise ValueError("Spearman correlation is undefined for constant ranks")
    value = float(np.dot(left_centered, right_centered) / denominator)
    if not math.isfinite(value):
        raise ValueError("Spearman correlation is non-finite")
    return value


def ndcg_at_k(
    query_embeddings: NDArray[np.floating],
    corpus_embeddings: NDArray[np.floating],
    query_ids: Sequence[str],
    corpus_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, float]],
    *,
    k: int = 10,
    exclude_identical_ids: bool = True,
) -> dict[str, Any]:
    """Calculate deterministic dense-retrieval nDCG@k over the complete corpus."""

    if k < 1 or k > len(corpus_ids):
        raise ValueError("k must be between one and the corpus size")
    if query_embeddings.shape[0] != len(query_ids):
        raise ValueError("query embeddings and identifiers must align")
    if corpus_embeddings.shape[0] != len(corpus_ids):
        raise ValueError("corpus embeddings and identifiers must align")
    if query_embeddings.shape[1] != corpus_embeddings.shape[1]:
        raise ValueError("query and corpus embedding dimensions must match")
    corpus_index = {identifier: index for index, identifier in enumerate(corpus_ids)}
    if len(corpus_index) != len(corpus_ids):
        raise ValueError("corpus identifiers must be unique")

    scores = np.asarray(query_embeddings, dtype=np.float32) @ np.asarray(
        corpus_embeddings, dtype=np.float32
    ).T
    ndcgs: list[float] = []
    relevant_in_top_k = 0
    for query_index, query_id in enumerate(query_ids):
        relevance = qrels.get(query_id)
        if not relevance:
            raise ValueError(f"missing relevance judgment for query {query_id}")
        query_scores = scores[query_index].copy()
        if exclude_identical_ids and query_id in corpus_index:
            query_scores[corpus_index[query_id]] = -np.inf
        top_indices = np.argpartition(query_scores, -k)[-k:]
        ranked_indices = sorted(
            (int(index) for index in top_indices),
            key=lambda index: (-float(query_scores[index]), corpus_ids[index]),
        )
        dcg = 0.0
        for rank, index in enumerate(ranked_indices, start=1):
            gain = float(relevance.get(corpus_ids[index], 0.0))
            if gain > 0.0:
                dcg += (2.0**gain - 1.0) / math.log2(rank + 1.0)
        ideal_gains = sorted((float(value) for value in relevance.values()), reverse=True)[:k]
        ideal_dcg = sum(
            (2.0**gain - 1.0) / math.log2(rank + 1.0)
            for rank, gain in enumerate(ideal_gains, start=1)
        )
        if ideal_dcg <= 0.0:
            raise ValueError(f"query {query_id} has no positive relevance judgment")
        ndcg = dcg / ideal_dcg
        ndcgs.append(ndcg)
        relevant_in_top_k += int(ndcg > 0.0)
    mean_ndcg = float(np.mean(np.asarray(ndcgs, dtype=np.float64)))
    return {
        f"ndcg_at_{k}": mean_ndcg,
        f"ndcg_at_{k}_x100": mean_ndcg * 100.0,
        "queries_scored": len(ndcgs),
        f"queries_with_relevant_in_top_{k}": relevant_in_top_k,
    }


def _encode_texts(
    session: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    batch_size: int,
) -> tuple[NDArray[np.float32], float]:
    batches: list[NDArray[np.float32]] = []
    started = perf_counter()
    for start in range(0, len(texts), batch_size):
        feeds = _feeds(tokenizer, session, texts[start : start + batch_size])
        batches.append(_embed(session, feeds))
    elapsed = perf_counter() - started
    if not batches:
        raise ValueError("cannot encode an empty text collection")
    embeddings = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
    if embeddings.shape[0] != len(texts) or not np.isfinite(embeddings).all():
        raise ValueError("encoder returned invalid embeddings")
    return embeddings, elapsed


def _fidelity(
    control: NDArray[np.floating], candidate: NDArray[np.floating]
) -> dict[str, float]:
    cosines = rowwise_cosine(control, candidate).astype(np.float64, copy=False)
    return {
        "mean_corresponding_embedding_cosine": float(np.mean(cosines)),
        "minimum_corresponding_embedding_cosine": float(np.min(cosines)),
    }


def _evaluate_sts(
    sessions: Mapping[str, Any],
    tokenizer: Any,
    configs: Mapping[str, Mapping[str, Any]],
    *,
    batch_size: int,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    per_variant: dict[str, dict[str, Any]] = {
        name: {"by_config": {}, "encode_wall_seconds": 0.0} for name in sessions
    }
    fidelity_parts: dict[str, list[NDArray[np.float32]]] = {
        comparison: [] for comparison, _, _ in QUALITY_COMPARISONS
    }
    for config_name, config in sorted(configs.items()):
        sentence1 = config["sentence1"]
        sentence2 = config["sentence2"]
        gold_scores = np.asarray(config["scores"], dtype=np.float64)
        texts = (*sentence1, *sentence2)
        embeddings: dict[str, NDArray[np.float32]] = {}
        for variant_name, session in sessions.items():
            encoded, elapsed = _encode_texts(
                session,
                tokenizer,
                texts,
                batch_size=batch_size,
            )
            embeddings[variant_name] = encoded
            per_variant[variant_name]["encode_wall_seconds"] += elapsed
            pair_count = len(sentence1)
            similarities = np.sum(encoded[:pair_count] * encoded[pair_count:], axis=1)
            correlation = spearman_correlation(gold_scores, similarities)
            per_variant[variant_name]["by_config"][config_name] = {
                "pairs": pair_count,
                "cosine_spearman": correlation,
                "cosine_spearman_x100": correlation * 100.0,
            }
        for comparison, control, candidate in QUALITY_COMPARISONS:
            fidelity_parts[comparison].append(
                rowwise_cosine(embeddings[control], embeddings[candidate])
            )
        if progress is not None:
            progress(f"Completed STS configuration {config_name} ({len(sentence1)} pairs).")

    for variant in per_variant.values():
        values = [item["cosine_spearman"] for item in variant["by_config"].values()]
        macro = float(np.mean(np.asarray(values, dtype=np.float64)))
        variant["macro_cosine_spearman"] = macro
        variant["macro_cosine_spearman_x100"] = macro * 100.0
    fidelity = {}
    for comparison, parts in fidelity_parts.items():
        cosines = np.concatenate(parts).astype(np.float64, copy=False)
        fidelity[comparison] = {
            "mean_corresponding_embedding_cosine": float(np.mean(cosines)),
            "minimum_corresponding_embedding_cosine": float(np.min(cosines)),
        }
    return {
        "task": "IndicCrosslingualSTS",
        "primary_score": "macro_cosine_spearman_x100",
        "configuration_count": len(configs),
        "pairs_per_configuration": INDIC_STS_ROWS_PER_CONFIG,
        "scores_by_variant": per_variant,
        "fidelity_by_comparison": fidelity,
    }


def _evaluate_retrieval(
    sessions: Mapping[str, Any],
    tokenizer: Any,
    data: Mapping[str, Any],
    *,
    batch_size: int,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    corpus_ids = data["corpus_ids"]
    query_ids = data["query_ids"]
    embeddings: dict[str, dict[str, NDArray[np.float32]]] = {}
    scores_by_variant: dict[str, dict[str, Any]] = {}
    for variant_name, session in sessions.items():
        corpus_embeddings, corpus_seconds = _encode_texts(
            session,
            tokenizer,
            data["corpus_texts"],
            batch_size=batch_size,
        )
        query_embeddings, query_seconds = _encode_texts(
            session,
            tokenizer,
            data["query_texts"],
            batch_size=batch_size,
        )
        embeddings[variant_name] = {
            "corpus": corpus_embeddings,
            "queries": query_embeddings,
        }
        scores_by_variant[variant_name] = {
            **ndcg_at_k(
                query_embeddings,
                corpus_embeddings,
                query_ids,
                corpus_ids,
                data["qrels"],
                k=10,
                exclude_identical_ids=True,
            ),
            "corpus_encode_wall_seconds": corpus_seconds,
            "query_encode_wall_seconds": query_seconds,
        }
        if progress is not None:
            progress(
                f"Completed ArguAna variant {variant_name} "
                f"({len(corpus_ids)} documents, {len(query_ids)} queries)."
            )

    fidelity = {}
    for comparison, control, candidate in QUALITY_COMPARISONS:
        corpus_fidelity = _fidelity(
            embeddings[control]["corpus"], embeddings[candidate]["corpus"]
        )
        query_fidelity = _fidelity(
            embeddings[control]["queries"], embeddings[candidate]["queries"]
        )
        combined_control = np.concatenate(
            (embeddings[control]["corpus"], embeddings[control]["queries"]), axis=0
        )
        combined_candidate = np.concatenate(
            (embeddings[candidate]["corpus"], embeddings[candidate]["queries"]), axis=0
        )
        fidelity[comparison] = {
            **_fidelity(combined_control, combined_candidate),
            "corpus": corpus_fidelity,
            "queries": query_fidelity,
        }
    return {
        "task": "ArguAna",
        "primary_score": "ndcg_at_10_x100",
        "corpus_rows": len(corpus_ids),
        "query_rows": len(query_ids),
        "qrel_rows": sum(len(items) for items in data["qrels"].values()),
        "unretrievable_qrel_targets": list(data["missing_qrel_targets"]),
        "unretrievable_qrel_target_count": len(data["missing_qrel_targets"]),
        "ignore_identical_ids": True,
        "scores_by_variant": scores_by_variant,
        "fidelity_by_comparison": fidelity,
    }


def quality_comparison_decision(
    *,
    control_sts_x100: float,
    candidate_sts_x100: float,
    control_retrieval: float,
    candidate_retrieval: float,
    sts_mean_cosine: float,
    retrieval_mean_cosine: float,
) -> dict[str, Any]:
    """Apply the predeclared downstream-quality thresholds to one comparison."""

    if control_retrieval <= 0.0:
        raise ValueError("control retrieval score must be positive")
    values = (
        control_sts_x100,
        candidate_sts_x100,
        control_retrieval,
        candidate_retrieval,
        sts_mean_cosine,
        retrieval_mean_cosine,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("quality comparison values must be finite")
    sts_loss = control_sts_x100 - candidate_sts_x100
    retrieval_relative_loss = (control_retrieval - candidate_retrieval) / control_retrieval
    gates = {
        "sts_absolute_loss_points_at_most_0_5": sts_loss <= 0.5 + 1e-12,
        "retrieval_relative_loss_at_most_0_01": retrieval_relative_loss <= 0.01 + 1e-12,
        "sts_mean_embedding_cosine_at_least_0_99": sts_mean_cosine >= 0.99,
        "retrieval_mean_embedding_cosine_at_least_0_99": retrieval_mean_cosine >= 0.99,
    }
    passed = all(gates.values())
    return {
        "sts_absolute_loss_points": sts_loss,
        "retrieval_relative_loss": retrieval_relative_loss,
        "sts_mean_corresponding_embedding_cosine": sts_mean_cosine,
        "retrieval_mean_corresponding_embedding_cosine": retrieval_mean_cosine,
        "gates": gates,
        "passed": passed,
        "status": "passed" if passed else "rejected-task-quality-gate",
    }


def _build_comparisons(sts: Mapping[str, Any], retrieval: Mapping[str, Any]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for comparison, control, candidate in QUALITY_COMPARISONS:
        comparisons[comparison] = {
            "control": control,
            "candidate": candidate,
            **quality_comparison_decision(
                control_sts_x100=sts["scores_by_variant"][control][
                    "macro_cosine_spearman_x100"
                ],
                candidate_sts_x100=sts["scores_by_variant"][candidate][
                    "macro_cosine_spearman_x100"
                ],
                control_retrieval=retrieval["scores_by_variant"][control]["ndcg_at_10"],
                candidate_retrieval=retrieval["scores_by_variant"][candidate]["ndcg_at_10"],
                sts_mean_cosine=sts["fidelity_by_comparison"][comparison][
                    "mean_corresponding_embedding_cosine"
                ],
                retrieval_mean_cosine=retrieval["fidelity_by_comparison"][comparison][
                    "mean_corresponding_embedding_cosine"
                ],
            ),
        }
    return comparisons


def evaluate_quality_verdict(result: Mapping[str, Any]) -> dict[str, Any]:
    """Combine the deterministic task gate with the required Arm64 BF16 capability check."""

    architecture = str(result["machine"]["architecture"]).lower()
    features = set((result["machine"].get("linux_cpu") or {}).get("features", []))
    if architecture not in {"aarch64", "arm64"}:
        return {
            "status": "needs-native-arm64-evaluation",
            "promotion_ready": False,
            "reason": "The task pipeline ran, but BF16 promotion requires native Arm64.",
        }
    if features and not features.intersection({"bf16", "svebf16"}):
        return {
            "status": "aborted-unsupported-hardware",
            "promotion_ready": False,
            "reason": "The native target does not advertise BF16 support.",
        }
    comparison = result["comparisons"]["fp32_bf16_vs_control"]
    if not comparison["passed"]:
        return {
            "status": "rejected-task-quality-gate",
            "promotion_ready": False,
            "promotion_scope": "fp32_bf16_fastmath",
            "reason": "FP32+BF16 failed at least one predeclared downstream-quality threshold.",
        }
    return {
        "status": "fp32-bf16-task-quality-gate-passed",
        "promotion_ready": True,
        "promotion_scope": "fp32_bf16_fastmath",
        "submitted_qint8_quality_status": result["comparisons"]["qint8_control_vs_fp32"][
            "status"
        ],
        "qint8_bf16_quality_status": result["comparisons"]["qint8_bf16_vs_control"][
            "status"
        ],
        "reason": (
            "FP32+BF16 passed the task gate; its five-run performance gate was completed "
            "separately. QInt8 conclusions remain limited to their recorded comparisons."
        ),
    }


def run_quality_evaluation(
    paths: ModelPaths,
    *,
    work_dir: Path,
    batch_size: int,
    threads: int,
    code_revision: str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Evaluate the four BF16 experiment variants on the fixed downstream task gate."""

    if batch_size < 1 or threads < 1:
        raise ValueError("batch_size and threads must be positive")
    data = load_quality_data(work_dir)
    if progress is not None:
        progress("Verified revision-pinned task data and source hashes.")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        use_fast=True,
    )
    variants = bf16_variants(paths)
    sessions: dict[str, Any] = {}
    variant_metadata: dict[str, Any] = {}
    for variant in variants:
        session, load_ms = _create_session(
            variant.model_path,
            threads=threads,
            session_config=variant.session_config,
        )
        sessions[variant.name] = session
        variant_metadata[variant.name] = {
            "model_kind": variant.model_kind,
            "session_config": dict(variant.session_config),
            **_model_metadata(variant.model_path, load_ms=load_ms),
        }

    sts = _evaluate_sts(
        sessions,
        tokenizer,
        data["sts"]["configs"],
        batch_size=batch_size,
        progress=progress,
    )
    retrieval = _evaluate_retrieval(
        sessions,
        tokenizer,
        data["retrieval"],
        batch_size=batch_size,
        progress=progress,
    )
    result: dict[str, Any] = {
        "schema_version": "armbench-quality/v1",
        "experiment": {
            "id": QUALITY_EXPERIMENT_ID,
            "parent": BF16_EXPERIMENT_ID,
            "origin": "agent-proposed-after-primary-source-and-license-review",
            "hypothesis": (
                "The BF16 session entry preserves same-artifact task quality within the "
                "predeclared STS, retrieval, and corresponding-embedding thresholds."
            ),
            "smallest_delta": (
                "Evaluate the four unchanged BF16 experiment variants; do not change either graph."
            ),
            "success_criteria": {
                "mean_corresponding_embedding_cosine_at_least": 0.99,
                "sts_absolute_loss_points_at_most": 0.5,
                "retrieval_relative_loss_at_most": 0.01,
            },
            "abort_criteria": (
                "Abort on source hash/schema/cardinality drift, non-finite metrics, or a native "
                "target without advertised BF16 support."
            ),
            "code_revision": code_revision or os.getenv("GITHUB_SHA") or "working-tree",
        },
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": "Apache-2.0",
        },
        "data_manifest": {
            "redistributed": False,
            "indic_crosslingual_sts": data["sts"]["manifest"],
            "arguana": data["retrieval"]["manifest"],
        },
        "configuration": {
            "batch_size": batch_size,
            "max_length": MAX_LENGTH,
            "truncation": True,
            "padding": "dynamic-per-batch",
            "pooling": "attention-mask-mean",
            "normalization": "row-wise-l2",
            "intra_op_threads": threads,
            "inter_op_threads": 1,
            "execution_provider": "CPUExecutionProvider",
            "official_mteb_leaderboard_result": False,
        },
        "machine": _machine_metadata(),
        "runtime": {
            "python": platform.python_version(),
        },
        "variants": variant_metadata,
        "tasks": {
            "indic_crosslingual_sts": sts,
            "arguana": retrieval,
        },
    }
    result["comparisons"] = _build_comparisons(sts, retrieval)
    result["verdict"] = evaluate_quality_verdict(result)
    return result


def render_quality_markdown(result: Mapping[str, Any]) -> str:
    """Render the quality evidence as a concise reviewer-facing report."""

    experiment = result["experiment"]
    sts = result["tasks"]["indic_crosslingual_sts"]
    retrieval = result["tasks"]["arguana"]
    lines = [
        f"# Experiment {experiment['id']}",
        "",
        f"- Generated: `{result['generated_at_utc']}`",
        f"- Parent: `{experiment['parent']}`",
        f"- Code revision: `{experiment['code_revision']}`",
        f"- Architecture: `{result['machine']['architecture']}`",
        "",
        "## Fixed task contract",
        "",
        (
            f"- IndicCrosslingualSTS: 12 x {sts['pairs_per_configuration']} pairs, "
            "macro cosine Spearman x 100."
        ),
        (
            f"- ArguAna: {retrieval['corpus_rows']:,} documents and "
            f"{retrieval['query_rows']:,} queries, nDCG@10 x 100."
        ),
        (
            "- ArguAna source limitation: "
            f"{retrieval['unretrievable_qrel_target_count']} pinned qrel targets are absent "
            "from the corpus and therefore score zero."
        ),
        f"- Tokenizer/model revision: `{result['source_model']['revision']}`.",
        "- This is a max-length-128 engineering gate, not an official MTEB leaderboard result.",
        "",
        "## Task scores",
        "",
        "| Variant | STS macro Spearman | ArguAna nDCG@10 |",
        "|---|---:|---:|",
    ]
    for name in (
        "fp32_control",
        "fp32_bf16_fastmath",
        "qint8_control",
        "qint8_bf16_fastmath",
    ):
        lines.append(
            f"| `{name}` | "
            f"{sts['scores_by_variant'][name]['macro_cosine_spearman_x100']:.4f} | "
            f"{retrieval['scores_by_variant'][name]['ndcg_at_10_x100']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Predeclared comparison gates",
            "",
            (
                "| Comparison | STS loss (points) | Retrieval relative loss | "
                "STS cosine | Retrieval cosine | Verdict |"
            ),
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for name, comparison in result["comparisons"].items():
        lines.append(
            f"| `{name}` | {comparison['sts_absolute_loss_points']:.4f} | "
            f"{comparison['retrieval_relative_loss'] * 100.0:.4f}% | "
            f"{comparison['sts_mean_corresponding_embedding_cosine']:.8f} | "
            f"{comparison['retrieval_mean_corresponding_embedding_cosine']:.8f} | "
            f"**{comparison['status']}** |"
        )
    lines.extend(
        [
            "",
            "## STS slices",
            "",
            "| Language pair | FP32 | FP32+BF16 | QInt8 | QInt8+BF16 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for config in sorted(sts["scores_by_variant"]["fp32_control"]["by_config"]):
        cells = []
        for name in (
            "fp32_control",
            "fp32_bf16_fastmath",
            "qint8_control",
            "qint8_bf16_fastmath",
        ):
            cells.append(
                sts["scores_by_variant"][name]["by_config"][config][
                    "cosine_spearman_x100"
                ]
            )
        lines.append(
            f"| `{config}` | {cells[0]:.4f} | {cells[1]:.4f} | "
            f"{cells[2]:.4f} | {cells[3]:.4f} |"
        )
    verdict = result["verdict"]
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"**{verdict['status']}** — {verdict['reason']}",
            "",
            "The data files are fetched from exact official MTEB task revisions, hash-checked, "
            "and not redistributed in this repository.",
            "",
        ]
    )
    return "\n".join(lines)


def write_quality_evaluation(result: Mapping[str, Any], output_dir: Path) -> dict[str, Path]:
    """Write machine-readable and reviewer-readable quality evidence."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "quality.json",
        "markdown": output_dir / "quality.md",
    }
    paths["json"].write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    paths["markdown"].write_text(
        render_quality_markdown(result),
        encoding="utf-8",
        newline="\n",
    )
    return paths
