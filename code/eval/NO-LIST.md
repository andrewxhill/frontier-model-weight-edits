# Score open naming

With no list of names in the prompt, the camera-trap edit raised balanced accuracy from 34.88% to 41.19% under strict naming, a gain of 6.31 percentage points. Under taxonomy-lenient scoring it rose from 47.78% to 49.73%, a gain of 1.94 points. The scores are in `results/check39-detail/no-list-summary.json`; the paper calls the camera-trap edit check 39.

The prompt is fixed:

> What animal is shown in this image? Answer the question using a single word or phrase.

`no_list_eval.py` uses the same system message, frame hashes, 25%-padded detector crop, chat template, float32, batch-one generation and greedy decoding as the list evaluator, including repetition penalty 1.1. It gives the model no option names, IDs or taxonomy. The manifest covers the same 1,611 test crops. The parser reads the whole free-text reply. Looped, truncated, empty or refused replies are recorded as faults. A reply that names no known animal is recorded as unmapped; explicit uncertainty, such as "I don't know", and empty answers count as missing.

## Run from the `code/` folder

```sh
python eval/no_list_eval.py --model path/to/model --manifest ../data/camera/manifest.json --images image-cache --output fresh-open-edited --device mps
python eval/no_list_scorer.py --edited fresh-open-edited/predictions.jsonl --output fresh-open-summary.json
```

To compare with the unedited model, run it into a separate new directory, then pass its predictions with `--baseline` alongside the edited predictions.

## Strict and taxonomy-lenient scoring

`no_list_scorer.py` uses the fixed aliases and parent-child relationships in `no-list-taxonomy.json`. Strict scoring normalises the whole phrase and applies the aliases, then compares the resulting category with the reference label, so cougar and puma count as mountain lion. Lenient scoring also accepts a declared ancestor or descendant: "bird" gets credit for any declared bird species. "Rodent" gets no credit for a rabbit, and "cat" gets no credit for a bobcat or mountain lion.

Normalisation turns hyphens and the dashes U+2010 to U+2015 into spaces. There is no fuzzy matching and no model judge. Unmapped and wrong answers are recorded separately from missing ones.
