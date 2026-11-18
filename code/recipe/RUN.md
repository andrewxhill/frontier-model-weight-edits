# Rebuild the camera-trap edit from public practice inputs

This recipe recomputes the camera-trap edit from the public base model, SigLIP and public practice crops. The rebuilt model scored 81.76% balanced accuracy on the 1,611-crop test and gave the same output as the released edit on every crop. Its weights are almost identical: 713 of 716 tensors match byte for byte, and three differ slightly. [Construction details](CONSTRUCTION.md) report those differences. To reproduce the released weights exactly, use `apply_edit.py` and the delta.

The practice crops are the images and labels the editing models could use. The 45 test cameras and the 204 tuning crops held back from the editors are not in these files. There are 23,447 practice crops from 57 cameras: 23,139 with a label and 308 without. The recipe's fixed selections use 2,449 crops from 2,440 frames. It recomputes the feature banks in their original order from the original weights and practice images. It does not read the released edited model, the delta, or any saved feature arrays or fitted coefficients.

## Install dependencies and fetch inputs

The reference build used Python 3.13.14 on Apple silicon (MPS) with float32. Other operating systems and backends are untested. Create a separate environment. The build requirements add SciPy, SentencePiece and protobuf to the nine packages in `../requirements.txt`; `installed-build-freeze.txt` records the resolved versions.

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

Run these from the `code/` folder. `prepare_base048` loads `lmms-lab/llava-onevision-qwen2-0.5b-ov` at revision `381d9947148efb1e58a577f451c05705ceec666e`, checks its hashes and saves it as a standard Transformers model. The camera model supplies activations from its 26-layer vision encoder. The original SigLIP model supplies the final vision block, normalisation and pooling head that turn those activations into image vectors, and the text tower that turns animal names into matching vectors. Its weights, configuration, prompt and tokens are checked against the reference metadata. The fetch script checks every frame's hash and the fixed selection.

## Build

```sh
python build_edit.py --base work/base --reference work/reference \
  --reference-metadata recipe/reference-metadata047.json \
  --practice ../data/camera/practice/crops.json --cameras ../data/camera/practice/cameras.json \
  --images work/practice-images --selections recipe/selections047.json \
  --condition recipe/condition047.json --output work/construction --device mps
```

Use a new output directory. The build writes `derivation-receipt.json` and a standard `derived-model` directory. It saves and reloads that model and records its state hash. The build never reads the released edit; compare against it afterwards.

Intermediate captures are tied to hashes of the code, inputs, model and runtime. `--resume` continues an unfinished build only when all of those still match. The builder refuses to overwrite an existing `derived-model` directory. If you change the environment or inputs, start a new build.

## Disk and memory

After installing the environments and downloading the models and images, keep at least 36.71 GB free for the build: 18.71 GB for outputs, 2 GB for temporary model writes and 16 GB of headroom. Allow more if other programs write to the same disk. A resumed build keeps earlier captures and adds a 1.72 GB memory-mapped copy of the base. The first capture stage, `T1` in the builder and selection file, processes 512 crops. The image processor expands them into 3,571 views (whole-image views and tiles, up to 37 per crop).

The reference build stayed under 30 GB of process memory, not counting GPU or unified-memory allocations, within a 32 GB ceiling. The builder has no memory or disk guard of its own, so watch both while it runs. The recipe makes 2,570 generations of up to 1,024 new tokens each, plus diagnostic and reference captures.
