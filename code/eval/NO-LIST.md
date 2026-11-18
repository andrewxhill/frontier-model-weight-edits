# Evaluate open naming

Without option lists, check 39 gained 6.31 percentage points under strict naming and 1.94 under taxonomy-lenient scoring. Balanced accuracy rose from 34.88% to 41.19% and from 47.78% to 49.73% respectively. Both rows were reproduced through this standalone path; the scores are in `results/check39-detail/no-list-summary.json`.

The prompt is fixed:

> What animal is shown in this image? Answer the question using a single word or phrase.

`no_list_eval.py` uses the same system message, full-frame hashes, 25%-padded detector crop, chat template, float32, batch-one generation and greedy decoder as the with-list evaluator, including repetition penalty 1.1. It supplies no option names, IDs or taxonomy to the model. The manifest covers the same 1,611 test crops. The parser evaluates the whole free-text reply without requiring it to match an option list. Looped, truncated, empty or refused replies are recorded as faults. A reply that cannot be mapped to a known name can still be an unmapped answer; explicit uncertainty and empty answers count as missing.

## Run from the `code/` folder

```sh
python eval/no_list_eval.py --model path/to/model --manifest ../data/camera/manifest.json --images image-cache --output fresh-open-edited --device mps
python eval/no_list_scorer.py --edited fresh-open-edited/predictions.jsonl --output fresh-open-summary.json
```

To compare with unedited, run the converted original model into a separate new directory, then pass its predictions with `--baseline` alongside the edited predictions.

## Strict and taxonomy-lenient scoring

`no_list_scorer.py` uses the fixed aliases and parent relationships in `no-list-taxonomy.json`. Strict scoring normalises the whole phrase and applies the declared aliases before comparing its category with the reference label. Thus cougar and puma map to mountain lion even under strict scoring. The lenient score also accepts declared ancestor/descendant compatibility.

Whole-phrase normalisation includes the original U+2010 through U+2015 dash range. There is no fuzzy matching. Bird and a declared bird descendant can receive lenient credit. Rabbit is not a rodent; domestic cat is not an ancestor of bobcat or mountain lion. Unknown, unmapped and wrong answers are recorded separately from missing answers. The scorer uses these fixed rules rather than a model judge.
