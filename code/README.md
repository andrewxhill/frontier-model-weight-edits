# Code

Run scripts from this folder. `pip install -r requirements.txt` installs the tested versions. Every script takes `--help`. From the repository root, `python -m unittest discover -s code/tests` runs the smoke tests.

| Path | Purpose |
|---|---|
| `apply_edit.py` | Rebuild a released model from the public base and its `delta/` files, checking every byte |
| `make_delta.py` | Write the delta and manifest for a released model |
| `tensor_identity.py` | Hashes of each tensor and of the whole model state, used by both |
| `eval/` | Camera-trap benchmark: fetch frames, generate answers, score ([eval/README.md](eval/README.md), [eval/NO-LIST.md](eval/NO-LIST.md)) |
| `island/` | Island test: fetch crops, generate answers from the 11-name lists, score |
| `summarise_results.py` | Recompute `results/summary.json` from saved predictions |
| `build_edit.py`, `recipe/` | Recompute the camera-trap edit from the base, public practice crops and SigLIP ([recipe/RUN.md](recipe/RUN.md)) |
| `construct/` | Fitting code for the island edit (`island_math.py`), the direct combined edit (`combined_fit.py`, `fit_weights.py`) and the LoRA model (`lora_core.py`) |
| `tests/` | Smoke tests: every script parses its arguments; saved predictions reproduce the headline scores |

## Construction settings

The projector is the small network that passes image features to the language model. A name hint is a projector output that pushes the model towards one animal name, written at two of the image-token positions the language model reads. Practice crops fit a model; selection crops choose between candidates; test crops are scored once.

**Island edit.** The edit keeps the unedited model's language weights and changes only the projector and 13 output rows: the letters A to K and the two rows for "The". Its projector starts from the camera-trap edit's projector with that edit's 58 camera-name hints removed, so it reuses the same hint mechanism. Ridge regressions (least-squares fits with a penalty that keeps the numbers small) map the vision features of 846 practice crops to a score for each name on the island lists. The fitted scores are written into projector units as name hints. The chosen edit uses hint magnitude 0.6, a shortened distinguishing name for each hint, and a practice-fitted correction to the answer-letter rows multiplied by 4. It was the best of sixteen candidates on the 426 selection crops.

**Direct combined edit.** Starting from the camera-trap edit, one joint ridge fit adds the island hints while asking for zero change on 2,095 camera practice crops. Eight candidates crossed four camera-to-island weight ratios with two hint magnitudes. Selection on 194 camera and 426 island crops chose ratio 0.1 and magnitude 0.6.

**LoRA model.** LoRA trains a small low-rank add-on to chosen weight matrices and then merges it in. Starting from the camera-trap edit: rank 8, alpha 16, on the query and value matrices of all 24 language blocks (540,672 trained parameters). One seeded pass (seed 20261118) over the same 2,902 practice crops, float32, one crop per step with eight-step accumulation, AdamW at a constant 0.0002, no weight decay, gradients clipped at 1. Each domain carries half the loss. The adapter was then merged into the weights.
