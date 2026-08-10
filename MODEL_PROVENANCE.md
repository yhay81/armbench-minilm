# Model provenance

ArmBench MiniLM downloads one public model artifact at benchmark time. No model
weights are committed to this repository.

| Field | Value |
|---|---|
| Model | `sentence-transformers/all-MiniLM-L6-v2` |
| Publisher | Sentence Transformers |
| Revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Source file | `onnx/model.onnx` |
| Upstream license | Apache-2.0 |
| Intended use | Sentence and short-paragraph embeddings |
| Upstream limit | Inputs beyond 256 word pieces are truncated by the model |
| Retrieval | `huggingface_hub.hf_hub_download` with the exact revision |

Source: <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>

The benchmark creates its INT8 model locally with ONNX Runtime dynamic
quantization. The derived file is temporary CI output and is not redistributed
from this source repository. Reports record the source revision, file sizes,
quantization parameters, runtime version, machine architecture, and runner image.

## Benchmark text

The short English sentences used for latency and output-fidelity checks were
written specifically for this project. They are not an evaluation dataset and
must not be interpreted as a general semantic-similarity benchmark.
