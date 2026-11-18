# Build from public practice inputs

The practice-input computation reproduced 81.76% balanced accuracy and every check-39 output on the 1,611-crop test. It produced a different state: 713 of 716 tensors match byte for byte, while three differ slightly. [Construction details](CONSTRUCTION.md) report those differences. Exact reconstruction of the released weights uses `apply_edit.py` and the delta.

The practice crops are the images and labels made available to the editing programs. Private tuning crops and the 45 test cameras are excluded from these manifests. The bundle contains 23,447 practice crops from 57 cameras, with 23,139 disclosed labels and 308 null labels. Fixed selections use 2,449 crops from 2,440 frames. The recipe recomputes feature banks in their recorded order from original weights and practice images. It does not read the released edited checkpoint, delta, old feature arrays or fitted coefficients.

## Install dependencies and fetch inputs

The completed build used Python 3.13.14 on Apple silicon MPS with float32. Create a separate environment. The build requirements add SciPy, SentencePiece and protobuf to the nine inference pins; `installed-build-freeze.txt` records the resolved versions. Other operating systems and backends remain untested.

```sh
python -m venv .venv-build
. .venv-build/bin/activate
python -m pip install -r recipe/requirements-build056.txt
mkdir -p work
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
python -m recipe.prepare_base048 --base-spec recipe/original-base-spec.json --output work/base
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='google/siglip-so400m-patch14-384', revision='9fdffc58afc957d1a03a25b10dba0329ab15c2a3', local_dir='work/reference', token=False)"
python recipe/fetch_practice049.py --practice ../data/camera/practice/crops.json --selections recipe/selections047.json --images work/practice-images
```

The base helper loads `lmms-lab/llava-onevision-qwen2-0.5b-ov` at revision `381d9947148efb1e58a577f451c05705ceec666e`. It verifies the original `c3706c18` state and tied aliases, then saves a standard Transformers model. The camera model supplies activations from its 26-layer vision encoder. The original SigLIP reference supplies the missing final vision block, normalisation and pooling head, which turn those activations into image vectors, plus the text tower that produces matching animal-name vectors. Its weights, configuration, prompt and tokens are checked against the reference metadata. Image fetches verify full-frame hashes and the fixed selection.

## One fixed construction

```sh
python build_edit.py --base work/base --reference work/reference \
  --reference-metadata recipe/reference-metadata047.json \
  --practice ../data/camera/practice/crops.json --cameras ../data/camera/practice/cameras.json \
  --images work/practice-images --selections recipe/selections047.json \
  --condition recipe/condition047.json --output work/construction --device mps
```

Use a new output directory. Completion writes `derivation-receipt.json` and a standard `derived-model` directory. The builder saves and reloads that model, and records its full base-anchored state hash. The released state is read only by the subsequent comparison, after construction finishes.

Intermediate captures are bound to code, inputs, model state and runtime hashes. `--resume` continues the same unfinished attempt only when those bindings match. An existing `derived-model` directory is refused. Changing environments or inputs requires a new attempt.

## Disk and memory

After installing environments and obtaining model inputs and images, reserve at least 36.71 GB free for construction. Allow additional space for other writers. The first feature-capture stage, named `T1` in the builder and selection file, processes 512 crops. The image processor expands these into 3,571 views: resized whole-image views and tiles, with a maximum of 37 views per crop. The reservation comprises 18.71 GB for persistent outputs, 2 GB for temporary model writes and 16 GB left free. A resume retains previous captures and adds a 1.72 GB base memory map.

The completed run used a process-tree memory monitor with a 32 GB ceiling and a stop threshold of 30 GB. Process resident memory does not include every GPU or unified-memory allocation. The standalone builder does not install a memory or disk guard, so monitor available memory and disk while it runs. The fixed recipe includes 2,570 generations, each allowing up to 1,024 new tokens, plus diagnostic and reference captures. It keeps the original view count.

The paper and `results/check39-detail/` report no-list and regional-list measurements alongside the label-informed camera-option condition.
