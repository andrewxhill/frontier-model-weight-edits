# Score the camera-trap benchmark

The camera-trap edit raised balanced accuracy from 64.09% to 81.76% on 1,611 crops from 45 cameras, with each camera's own list of animal names in the prompt. These commands generate fresh answers from any model and score them the same way.

The manifest lists, for each crop, the public LILA frame URL, the frame's SHA256 hash, the detector box, the camera's list of names and the reference label. Each camera's list holds the animals labelled at that camera, so it tells the model which animals that camera sees. [NO-LIST.md](NO-LIST.md) scores the same crops with no list. Images are fetched separately. The frames come from [Island Conservation](https://lila.science/datasets/island-conservation-camera-traps/) and [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps/), both under CDLA-Permissive terms.

## Run from the `code/` folder

Install the declared dependencies, then run:

```sh
python eval/fetch_images.py --manifest ../data/camera/manifest.json --images image-cache
python eval/eval.py --model path/to/model --manifest ../data/camera/manifest.json --images image-cache --output fresh-edited --device mps
python -m recipe.prepare_base048 --base-spec recipe/original-base-spec.json --output base-model
python eval/eval.py --model base-model --manifest ../data/camera/manifest.json --images image-cache --output fresh-baseline --device mps
python eval/scorer.py --edited fresh-edited/predictions.jsonl --baseline fresh-baseline/predictions.jsonl --output paired-summary.json
```

`base-model` is the unedited model, converted by `recipe/prepare_base048.py` from the pinned revision its publisher released. Each evaluation needs a new output directory. The evaluator hashes the local model files, verifies every full frame before cropping, and loads the processor and chat template saved with the model.

## Crop and decoding settings

The evaluator uses float32, batch one, left tokeniser padding, 25% padding around each box, no EXIF rotation and the processor's own resizing. Generation is greedy, with 1,024 maximum new tokens, one beam, the cache enabled and repetition penalty 1.1. The end, start and padding tokens and the top-k default are checked against the manifest. Each crop also gets a fixed random seed derived from the SHA256 of its image ID and box index.

The parser records refusals, loops, truncated replies and empty replies as faults. It accepts a letter or an animal name only when it points to exactly one option on the list. A letter and a name that agree are accepted; a letter and a name that conflict stay unresolved. The model is asked once per crop.

## Scores and outputs

The score averages class recall (the share of each animal named correctly) within each camera, then averages the 45 camera scores. Missing answers count as wrong. The paired 95% interval resamples the 45 per-camera changes 2,000 times with seed 20261112. It covers variation between cameras for these fixed weights, not variation between separate editing attempts.

`manifest-schema.json` describes the manifest. `manifest.py` checks crop and camera coverage, hashes and the public URL for each frame. LILA withholds Island Conservation images that show people, and no test crop has a HUMAN label.

This evaluator gives each model a list of names. [NO-LIST.md](NO-LIST.md) gives the commands for open naming, with no list. Saved predictions for every released model are in `results/camera/`.
