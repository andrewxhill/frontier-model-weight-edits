# Specialist models for Galápagos animals

Global wildlife classifiers are built to cover everywhere, so they miss animals that live in only one place. Google's SpeciesNet, a camera-trap classifier with more than 2,000 labels, has no label for the Galápagos land iguana, the marine iguana or the Galápagos mockingbirds. AI research agents built and tested specialist models for them on one desktop machine, starting from a small public vision-language model.

**Explainer:** [https://&lt;gh-user&gt;.github.io/&lt;repo&gt;/](https://<gh-user>.github.io/<repo>/) · **Paper:** [paper/paper.pdf](paper/paper.pdf) · **Models:** [Hugging Face](#models)

## Results

The study has three parts, all built on LLaVA-OneVision Qwen2 0.5B:

- **An island edit** for Galápagos animals SpeciesNet cannot name. Mockingbird versus brown noddy rose from 70.83% to 98.33%, and the share of invasive rats found from 2.96% to 81.23%.
- **A general camera-trap edit.** Camera-trap accuracy rose from 64.09% to 81.76%.
- **One model for both jobs**, built two ways from the camera-trap edit and the same 2,902 practice photos. A LoRA fine-tune scored 87.58% to 93.33% on the island contrasts and 79.22% on camera traps. A direct edit scored 80.70% on camera traps, 1.48 points above the LoRA model, a gap too small to measure (95% interval −2.54 to +5.80), but lost most of the island gains. The direct edit took about an hour to build and the LoRA model about three hours, on one Mac Studio.

The edits change the model's weights directly, with no training run; the LoRA model is a standard fine-tune. Each model is shown a cropped photo and a list of names, and answers with one name: 11 names on the island test, and each camera's own list of 2 to 12 names on the camera-trap test.

**Island animals.** 697 test crops from places not used to build any model. The table scores the 408 natives, look-alikes and rats; the other 289 are control animals such as cats, goats and coati. Each native-versus-look-alike score averages two shares, natives named as natives and look-alikes named as look-alikes, with each photo location counting equally. An answer naming either native iguana counts as native, and both iguana contrasts use the same 57 green iguanas, so a model that rarely names green iguanas scores the same on both.

| Model | Mockingbird vs brown noddy | Land iguana vs green iguana | Marine iguana vs green iguana | Invasive rats found |
|---|---:|---:|---:|---:|
| Island edit | **98.33%** | 81.22% | 80.37% | 81.23% |
| One model for both jobs (LoRA) | 93.33% | **87.58%** | **91.69%** | **88.20%** |
| One model for both jobs (direct edit) | 90.83% | 55.49% | 55.49% | 17.70% |
| Camera-trap edit | 91.67% | 51.83% | 51.83% | 13.75% |
| Unedited model | 70.83% | 50.00% | 50.00% | 2.96% |
| SpeciesNet | 3.33% | 12.44% | 12.44% | 8.11% |

SpeciesNet has no label for these natives, so it never names them correctly. Its standard output, after its geographic filter, also rarely names the brown noddy or the green iguana, which puts its scores below the 50% a coin would get.

**Camera traps.** 1,611 crops from 45 cameras in two public LILA datasets. Balanced accuracy gives each camera, and each animal class within a camera, equal weight.

| Model | Balanced accuracy | Change from unedited, 95% interval |
|---|---:|---|
| Camera-trap edit | **81.76%** | +17.67 points (12.41 to 23.61) |
| One model for both jobs (direct edit) | 80.70% | +16.61 points (11.34 to 22.62) |
| One model for both jobs (LoRA) | 79.22% | +15.13 points (9.25 to 21.24) |
| Island edit | 72.14% | +8.05 points (3.01 to 13.93) |
| Unedited model | 64.09% | |

Each native group's test photos come from one iNaturalist place, so the island scores describe those photos rather than the whole archipelago.

`results/summary.json` holds every number above and more. `python code/summarise_results.py --output results/summary.json` recomputes it from the saved per-crop predictions in `results/`.

## Models

| Model | What it is for | Hugging Face |
|---|---|---|
| Island edit | Galápagos natives and invasive rats | [&lt;hf-user&gt;/galapagos-island-edit](https://huggingface.co/<hf-user>/galapagos-island-edit) |
| One model for both jobs (LoRA) | Camera traps and Galápagos animals | [&lt;hf-user&gt;/camera-island-lora](https://huggingface.co/<hf-user>/camera-island-lora) |
| Camera-trap edit | Camera-trap option lists | [&lt;hf-user&gt;/camera-trap-edit](https://huggingface.co/<hf-user>/camera-trap-edit) |
| One model for both jobs (direct edit) | The direct-edit comparison | [&lt;hf-user&gt;/camera-island-direct-edit](https://huggingface.co/<hf-user>/camera-island-direct-edit) |

Each repository holds full float32 weights that load with `from_pretrained`, plus a small `delta/` folder that rebuilds those weights from the public base model.

## Rebuild a model from the public base

Install the pinned packages, then apply a model's delta to the public base. The delta holds exact replacement values for the tensors that changed: the projector, a few rows of the output head and, for the LoRA model, the merged attention query and value matrices.

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r code/requirements.txt
hf download <hf-user>/galapagos-island-edit delta/delta.safetensors delta/delta_manifest.json --local-dir island
cd code
python apply_edit.py --base lmms-lab/llava-onevision-qwen2-0.5b-ov \
  --revision 381d9947148efb1e58a577f451c05705ceec666e \
  --delta ../island/delta/delta.safetensors --manifest ../island/delta/delta_manifest.json \
  --output ../island-model
```

`apply_edit.py` converts the base to the standard Transformers layout, checks the base and every replacement against SHA256 hashes, and refuses any mismatch. Each rebuild below reproduced the released state hash exactly:

| Model | Delta size | Released state SHA256 |
|---|---:|---|
| Camera-trap edit | 7.4 MB | `846cdbb58f3ca04c9ccda139d6e2c412921519f11da6e72a9a268480800b9881` |
| Island edit | 7.4 MB | `04d806ca47d3ccc4beb1129d8d108493170e0c37b85232bce433aabde1a09bcd` |
| Direct combined edit | 7.4 MB | `ccf2ebea68f3113bbe22fca40350fd6ffdb6c6cabf30b8f158d0cd84e3b83f39` |
| LoRA model | 91 MB | `3aaba90e481f5a73c45d05696e339293bb7a9f70313ff64d043339171fa56188` |

`make_delta.py` writes a delta and manifest from any released model directory.

**How the weights were made.** `code/recipe/` recomputes the camera-trap edit from the base, public practice crops and the SigLIP reference model; [code/recipe/RUN.md](code/recipe/RUN.md) gives the commands. `code/construct/` holds the fitting code for the other three models: the closed-form fits behind the island and direct combined edits, and the rank-8 LoRA layer with its merge. The LoRA model trained one seeded pass over 2,902 practice crops (2,095 camera, 807 island), on the query and value matrices of all 24 language blocks, at rank 8, alpha 16 and learning rate 0.0002 with AdamW. The paper's methods section gives every setting.

## Score a model

**Camera traps.** Fetch the 1,611 public frames, generate answers and score them:

```sh
cd code
python eval/fetch_images.py --manifest ../data/camera/manifest.json --images image-cache
python eval/eval.py --model ../island-model --manifest ../data/camera/manifest.json --images image-cache --output camera-run --device mps
python eval/scorer.py --edited camera-run/predictions.jsonl --output camera-scores.json
```

**Island animals.** Fetch the crops, ask the model and score the answers:

```sh
python island/fetch_crops.py --manifest ../data/island/test.jsonl --output island-crops
python island/predict.py --model ../island-model --manifest ../data/island/test.jsonl --crops island-crops --output island-predictions.jsonl --device mps
python island/score.py --manifest ../data/island/test.jsonl --predictions island-predictions.jsonl --output island-scores.json
```

The island prompt shows 11 names, lettered A to K, and asks for the animal's name. The iguana list is: Galapagos land iguana, Santa Fe land iguana, Galapagos pink land iguana, marine iguana, common green iguana, black rat, house mouse, domestic cat, domestic goat, pig, European rabbit. All three lists are in [code/island/menus.json](code/island/menus.json). Decoding is greedy, float32, with repetition penalty 1.1.

`python -m unittest discover -s code/tests` checks that every script parses its arguments and that the saved predictions reproduce the headline scores.

## Data

The repository lists crop IDs, source URLs, boxes, labels and splits. It holds no images; the fetch scripts download them from their public sources and check each file's SHA256 hash.

| Source | Used for | Licence |
|---|---|---|
| [Island Conservation Camera Traps](https://lila.science/datasets/island-conservation-camera-traps/), via LILA | Camera-trap benchmark, practice crops, island controls | CDLA-Permissive-1.0 |
| [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps/), via LILA | Camera-trap benchmark and practice crops | CDLA-Permissive-1.0 |
| [iNaturalist](https://www.inaturalist.org/) research-grade observations | Galápagos natives, green iguanas, rats and birds | Each photo's own licence: CC0, CC-BY or CC-BY-SA, recorded per row with its attribution |

The paper also tested the camera-trap edit with a regional species list on [Felidae Conservation Fund](https://lila.science/datasets/felidae-conservation-fund/) images from LILA (CDLA-Permissive-1.0). That crop list is not in this repository.

Splits are by location, with no location shared between practice, selection and test. The camera-trap test uses 45 cameras that no practice crop comes from. The Galápagos natives in the island practice, selection and test sets come from three different iNaturalist places (IDs 49264, 29827 and 49265). [data/README.md](data/README.md) describes every file and field.

## Credits

Claude Fable 5.2, Claude Opus 5.5 and GPT-6 Astra Pro edited the weights for the general camera-trap edit, working in turn. The Codex research assistant (OpenAI GPT) built and evaluated the island edit, the direct combined edit and the LoRA comparison.

## Citation and licence

Cite with [CITATION.cff](CITATION.cff):

> Andrew Hill (2026). *Direct weight edits by an AI editor: repairing camera-trap answers and naming island animals a camera-trap classifier lacks.* https://&lt;gh-user&gt;.github.io/&lt;repo&gt;/paper.pdf

The code is licensed under [Apache-2.0](LICENSE). The model weights are Apache-2.0, as is their base. The island edit and both one-model-for-both-jobs models were fitted on iNaturalist photos, and iNaturalist's terms forbid using its data to train AI for commercial purposes. Data manifests keep their sources' terms. [LICENSES.md](LICENSES.md) lists every upstream licence and what it requires.
