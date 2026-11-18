# Evaluate the with-list benchmark

Check 39 improved camera-balanced accuracy from 64.09% to 81.76% on the fixed 1,611 crops from 45 cameras. These commands generate fresh replies and score that condition.

The manifest contains public LILA frame URLs, full-frame SHA256 hashes, normalised detector boxes, camera-specific option lists and reference labels. The option lists were built from labels at each camera. Images are fetched separately. The source datasets are [Island Conservation](https://lila.science/datasets/island-conservation-camera-traps/) and [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps/), both under CDLA-Permissive terms.

## Run from the `code/` folder

Install the declared dependencies, then run:

```sh
python eval/fetch_images.py --manifest ../data/camera/manifest.json --images image-cache
python eval/eval.py --model path/to/model --manifest ../data/camera/manifest.json --images image-cache --output fresh-edited --device mps
python -m recipe.prepare_base048 --base-spec recipe/original-base-spec.json --output base-model
python eval/eval.py --model base-model --manifest ../data/camera/manifest.json --images image-cache --output fresh-baseline --device mps
python eval/scorer.py --edited fresh-edited/predictions.jsonl --baseline fresh-baseline/predictions.jsonl --output paired-summary.json
```

`base-model` must be the original model converted by the supplied tools from the pinned maker revision. Each evaluation requires a new output directory, hashes the local model files and verifies every full frame before cropping. It loads the saved processor and chat template with local model files.

## Fixed crop and decoding settings

The evaluator uses float32, batch one, left tokeniser padding, 25% box padding, no EXIF transformation and native processor resizing. Generation is greedy, with 1,024 maximum new tokens, one beam, cache enabled and repetition penalty 1.1. EOS/BOS/PAD and top-k defaults are checked against the manifest. The seed is derived from the SHA256 of the image ID and box index, without a dataset prefix, as in the original run.

The parser records refusal, loop, truncation and empty-output faults. It accepts a letter or animal name only when the presented option is unambiguous. An agreeing letter/name pair is accepted; a conflicting pair stays unresolved. The evaluation makes no extra attempt to obtain an answer.

## Scores and outputs

The primary score averages class recall within each camera, then averages the 45 camera scores. Missing answers count as incorrect. The paired interval resamples the 45 camera effects 2,000 times with seed 20261112. It measures camera uncertainty for these fixed weights; separate editing sessions are a different source of variation.

`manifest-schema.json` describes the manifest structure. `manifest.py` checks crop and camera coverage, hashes, public URL mapping and the Island Conservation HUMAN-annotation check. LILA removes HUMAN image bytes from that dataset; missing HUMAN images must not be fetched from another source.

This evaluator runs the with-list condition. [NO-LIST.md](NO-LIST.md) gives the separate open-naming commands. Saved predictions for every released model are in `results/camera/`.
