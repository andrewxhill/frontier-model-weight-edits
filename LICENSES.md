# Licences

This repository's code is Apache-2.0 ([LICENSE](LICENSE)). The released models and the data manifests derive from the sources below, each read on 18 November 2026.

| Source | Licence | Read at | What this release redistributes | Conditions |
|---|---|---|---|---|
| LLaVA-OneVision Qwen2 0.5B, revision `381d9947` | Apache-2.0 | [model card at that revision](https://huggingface.co/lmms-lab/llava-onevision-qwen2-0.5b-ov/blob/381d9947148efb1e58a577f451c05705ceec666e/README.md) | Edited and fine-tuned weights | Include the licence, state the changes, keep notices |
| Qwen2-0.5B (language model inside LLaVA-OneVision) | Apache-2.0 | [LICENSE file](https://huggingface.co/Qwen/Qwen2-0.5B/raw/main/LICENSE) | Part of the weights above | As above; no trademark grant |
| SigLIP so400m-patch14-384 (vision encoder, and the reference model for fitting) | Apache-2.0 | [model card](https://huggingface.co/google/siglip-so400m-patch14-384), [big_vision LICENSE](https://raw.githubusercontent.com/google-research/big_vision/main/LICENSE) | Vision weights inside the models; no separate SigLIP files | As above |
| SpeciesNet 4.0.3a | Code: Apache-2.0. Weights: not redistributed | [LICENSE file](https://raw.githubusercontent.com/google/cameratrapai/main/LICENSE) | Nothing. Only its per-crop answers and scores appear in `results/` | None for scores |
| MegaDetector v5a | MIT | [LICENSE file](https://raw.githubusercontent.com/agentmorris/MegaDetector/main/LICENSE) | Detector boxes in the manifests | None for boxes |
| Island Conservation Camera Traps (LILA) | CDLA-Permissive-1.0 | [dataset page](https://lila.science/datasets/island-conservation-camera-traps/) | Image IDs, URLs, boxes and labels; no images | Give the licence text or link with published data; keep the provider's credit. LILA withholds images of people, and no manifest row has a human label |
| Caltech Camera Traps (LILA) | CDLA-Permissive-1.0 | [dataset page](https://lila.science/datasets/caltech-camera-traps/) | Image IDs, URLs, boxes and labels; no images | As above; the page asks users to cite Beery, Van Horn and Perona (ECCV 2018) |
| Felidae Conservation Fund (LILA) | CDLA-Permissive-1.0 | [dataset page](https://lila.science/datasets/felidae-conservation-fund/) | Scores only, in `results/check39-detail/` | As above; the page asks users to cite Granados et al., Wildlife Biology 2024 |
| CDLA-Permissive-1.0 | — | [licence text](https://cdla.dev/permissive-1-0/) | — | Section 3.4: results of using the data carry no obligations or restrictions |
| iNaturalist observations | Each photo's own licence: CC0, CC-BY or CC-BY-SA | [terms of use](https://www.inaturalist.org/pages/terms) (archived copy of the 11 July 2023 revision); [help page on photo use](https://help.inaturalist.org/en/support/solutions/articles/151000169918-can-i-use-the-photos-and-sounds-that-are-posted-on-inaturalist-) | Observation and photo IDs, URLs, boxes, labels, and each photo's licence and credit line; no images | Credit each photo when you use it. The terms of use say: "Users may not use any iNaturalist data for training artificial intelligence, machine learning models, large language models, or similar networks, algorithms, or systems for commercial purposes." |
| Charles Darwin Foundation Galápagos species checklist | Site footer: CC BY-NC-SA 4.0 | [checklist](https://datazone.darwinfoundation.org/en/checklist) | Nothing; the checklist only informed which species names went on the menus | None |

## What this means for the models

- **Camera-trap edit** uses only LILA data and Apache-2.0 models. It is released under Apache-2.0.
- **Island edit, direct combined edit and LoRA model** were fitted on iNaturalist photos, for non-commercial research. iNaturalist's terms prohibit using its data to train AI models for commercial purposes. Their model cards carry Apache-2.0, inherited from the base, with this restriction stated beside it.
