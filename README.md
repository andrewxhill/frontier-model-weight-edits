# Can weight-space programming surface the domain expertise hidden in a small vision-language model? Evidence from camera traps and Galápagos species

Frontier models edited a free vision model so that it names Galápagos animals that a purpose-built camera-trap classifier cannot. On held-out photos, the edited model told Galápagos mockingbirds from brown noddies with 98.33% balanced accuracy, up from 70.83% before the edit. Google's SpeciesNet classifier scored 3.33%: it has more than 2,000 labels and none for the Galápagos land iguana, the marine iguana or the Galápagos mockingbirds. On each island contrast, balanced accuracy averages two shares, natives named correctly and look-alikes named correctly, so guessing scores 50%. The small model itself was edited and tested on one desktop computer, at no compute cost.

The model is LLaVA-OneVision, a vision-language model (it reads an image and answers in text) with half a billion parameters. Its vision encoder comes from SigLIP, a model trained to place images and their text descriptions close together, so the encoder's image features can be compared directly with SigLIP's text features for each animal name. The knowledge was already inside it. On an early sample of wild photos, gathered before the island test set, that comparison, with no labelled examples, told Galápagos mockingbirds from other birds at 97.66% and native iguanas from the introduced green iguana at 79.94%. The model's typed answers called every iguana native and scored 50.00%.

Frontier models brought that knowledge into the answers by changing the weights (the numbers the model is made of) by calculation, with no training run. Two kinds of change did the work, and both mattered. The first is in the projector, the small network that passes image features to the language model. It now also writes an animal-name hint: a score for each name on the list, computed from the image, placed where the language model reads it. The hints carry what the image encoder sees into the answer. For the Galápagos animals, those scores were solved as ridge regressions (least-squares fits with a penalty that keeps the numbers small) on labelled practice photos. The second corrects the output rows (the numbers that score each possible next token) so the model gives that answer cleanly. In the camera-trap edit the model rambled ("B. The image shows a fox…"), so the editors set the rows for "The" to zero and scaled the end-of-answer row by 1.5. Putting those three rows back drops the edit's camera-trap score from 81.76% to 72.03%. In the island edit, hints alone raised the selection score from 39.38 to 43.41, and hints with corrected rows for the answer letters and for "The" reached 71.21; the island test does not separate the two on held-out photos.

**Explainer:** [https://andrewxhill.com/frontier-model-weight-edits/](https://andrewxhill.com/frontier-model-weight-edits/) · **Paper:** [paper/paper.pdf](paper/paper.pdf) · **Models:** [Hugging Face](#models)

## What was built

Every model starts from LLaVA-OneVision Qwen2 0.5B.

- A general camera-trap edit. Claude Fable 5.2, Claude Opus 5.5 and GPT-6 Astra Pro took turns reading what the weights did and writing new values. Camera-trap balanced accuracy rose from 64.09% to 81.76%.
- An island edit for the Galápagos animals SpeciesNet has no label for. The Codex research assistant (OpenAI GPT) solved it as ridge regressions and wrote it into the weights. Green iguanas named correctly went from 0 of 57 to 36 of 57, and the share of invasive rats found from 2.96% to 81.23%.
- One model for both jobs. The Codex research assistant (OpenAI GPT) built it two ways from the camera-trap edit and the same 2,902 practice photos, and measured both. A direct edit took about an hour and held camera traps at 80.70%, but lost most of the island gains. A standard LoRA fine-tune (a small add-on to the weights, trained by gradient descent) took about three hours and scored 87.58% to 93.33% on the island contrasts while holding camera traps at 79.22%. The two camera scores match within noise (direct edit 1.48 points higher, 95% interval −2.54 to +5.80).

## Results

Each model is shown a cropped photo and a list of names, and answers with one name: 11 names on the island test, and each camera's own list of 2 to 12 names on the camera-trap test.

| Model | Mockingbird vs brown noddy | Land iguana vs green iguana | Marine iguana vs green iguana | Invasive rats found |
|---|---:|---:|---:|---:|
| Island edit | **98.33%** | 81.22% | 80.37% | 81.23% |
| One model for both jobs (LoRA) | 93.33% | **87.58%** | **91.69%** | **88.20%** |
| One model for both jobs (direct edit) | 90.83% | 55.49% | 55.49% | 17.70% |
| Camera-trap edit | 91.67% | 51.83% | 51.83% | 13.75% |
| Unedited model | 70.83% | 50.00% | 50.00% | 2.96% |
| SigLIP alone | 98.33% | 78.29% | 77.45% | 46.86% |
| SpeciesNet | 3.33% | 12.44% | 12.44% | 8.11% |
| BioCLIP 2 | 100.00% | 94.49% | 94.48% | 83.29% |

Bold marks the best of the five LLaVA models in each column. SigLIP alone, SpeciesNet and BioCLIP 2 are reference models, run as released.

SpeciesNet has no label for these natives, so it never names them correctly. Its standard output, after its geographic filter (which swaps a species not known from the photo's country for a broader label), also rarely names the brown noddy or the green iguana, which puts its scores below 50%. SpeciesNet was scored on the island test only.

**Island animals.** 697 test crops from places not used to build any model. The table scores the 408 natives, look-alikes and rats; the other 289 are control animals such as cats, goats and coati. Each native-versus-look-alike score averages the two shares with each photo location counting equally. An answer naming either native iguana counts as native, and both iguana contrasts use the same 57 green iguanas, so a model that rarely names green iguanas scores the same on both.

**Camera traps.** 1,611 crops from 45 cameras in two public datasets from LILA, a library of camera-trap images. Balanced accuracy here gives each camera, and each animal class within a camera, equal weight. Each camera's list holds the animals labelled at that camera, so the list tells the model which animals that camera sees; the paper also reports scores with no list and with a regional species list.

| Model | Balanced accuracy | Change from unedited, 95% interval |
|---|---:|---|
| Camera-trap edit | **81.76%** | +17.67 points (12.41 to 23.61) |
| One model for both jobs (direct edit) | 80.70% | +16.61 points (11.34 to 22.62) |
| One model for both jobs (LoRA) | 79.22% | +15.13 points (9.25 to 21.24) |
| Island edit | 72.14% | +8.05 points (3.01 to 13.93) |
| Unedited model | 64.09% | |
| SigLIP alone | 82.79% | |
| BioCLIP 2 | 84.07% | +19.98 points (13.76 to 26.75) |

**BioCLIP 2.** A free, general model with half a billion parameters, edited by frontier AI with no training run, scores 81.76% on the camera traps, level within noise with BioCLIP 2 at 84.07% (BioCLIP 2 higher by 2.31 points, 95% interval −1.58 to +6.38). BioCLIP 2 is the open image model built to name species, trained on photos across the tree of life; it ran as released on the same crops and lists. On the island test it scores 94.49% and 94.48% on the iguanas, 100.00% on the birds and 83.29% on rats, having trained on 399 of the 684 iNaturalist photos in that test. It is a stronger image encoder than the SigLIP one inside this model, and putting it inside the language model is the next experiment. [results/bioclip2/](results/bioclip2/) has the full comparison.

`results/summary.json` holds every number above except the BioCLIP 2 rows, and more. `python code/summarise_results.py --output results/summary.json` recomputes it from the saved per-crop predictions in `results/`.

## Models

| Model | What it is for | Hugging Face |
|---|---|---|
| Island edit | Galápagos natives and invasive rats | [andrewxhill/galapagos-island-edit](https://huggingface.co/andrewxhill/galapagos-island-edit) |
| One model for both jobs (LoRA) | Camera traps and Galápagos animals | [andrewxhill/camera-island-lora](https://huggingface.co/andrewxhill/camera-island-lora) |
| Camera-trap edit | Camera-trap option lists | [andrewxhill/camera-trap-edit](https://huggingface.co/andrewxhill/camera-trap-edit) |
| One model for both jobs (direct edit) | The direct-edit comparison | [andrewxhill/camera-island-direct-edit](https://huggingface.co/andrewxhill/camera-island-direct-edit) |

Each repository holds full float32 weights that load with `from_pretrained`, plus a small `delta/` folder that rebuilds those weights from the public base model.

## Rebuild a model from the public base

Each Hugging Face repository has a `delta/` folder: the exact new values for the tensors (weight arrays) that changed. These are the projector, a few rows of the output head and, for the LoRA model, the merged attention query and value matrices. Install the pinned packages, download a delta and apply it to the public base:

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r code/requirements.txt
hf download andrewxhill/galapagos-island-edit delta/delta.safetensors delta/delta_manifest.json --local-dir island
cd code
python apply_edit.py --base lmms-lab/llava-onevision-qwen2-0.5b-ov \
  --revision 381d9947148efb1e58a577f451c05705ceec666e \
  --delta ../island/delta/delta.safetensors --manifest ../island/delta/delta_manifest.json \
  --output ../island-model
```

`apply_edit.py` converts the base to the standard Transformers layout, checks the base and every replacement against SHA256 hashes, and refuses any mismatch. A correct rebuild ends with the state hash in this table:

| Model | Delta size | Released state SHA256 |
|---|---:|---|
| Camera-trap edit | 7.4 MB | `846cdbb58f3ca04c9ccda139d6e2c412921519f11da6e72a9a268480800b9881` |
| Island edit | 7.4 MB | `04d806ca47d3ccc4beb1129d8d108493170e0c37b85232bce433aabde1a09bcd` |
| Direct combined edit | 7.4 MB | `ccf2ebea68f3113bbe22fca40350fd6ffdb6c6cabf30b8f158d0cd84e3b83f39` |
| LoRA model | 91 MB | `3aaba90e481f5a73c45d05696e339293bb7a9f70313ff64d043339171fa56188` |

`make_delta.py` writes a delta and manifest from any released model directory.

## Recompute the weights

`code/recipe/` recomputes the camera-trap edit from the base, public practice crops and the SigLIP model; [code/recipe/RUN.md](code/recipe/RUN.md) gives the commands. `code/construct/` holds the fitting code for the other three models: the closed-form fits behind the island and direct combined edits, and the rank-8 LoRA layer with its merge. The LoRA model trained for one seeded pass over 2,902 practice crops (2,095 camera, 807 island), on the query and value matrices of all 24 language blocks, at rank 8, alpha 16 and learning rate 0.0002 with AdamW. [code/README.md](code/README.md) and the paper's methods section give every setting.

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

From the repository root, `python -m unittest discover -s code/tests` checks that every script parses its arguments and that the saved predictions reproduce the headline scores.

## Data

The repository lists crop IDs, source URLs, boxes, labels and splits. It holds no images; the fetch scripts download them from their public sources and check each file's SHA256 hash.

| Source | Used for | Licence |
|---|---|---|
| [Island Conservation Camera Traps](https://lila.science/datasets/island-conservation-camera-traps/), via LILA | Camera-trap benchmark, practice crops, island controls | CDLA-Permissive-1.0 |
| [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps/), via LILA | Camera-trap benchmark and practice crops | CDLA-Permissive-1.0 |
| [iNaturalist](https://www.inaturalist.org/) research-grade observations | Galápagos natives, green iguanas, rats and birds | Each photo's own licence: CC0, CC-BY or CC-BY-SA, recorded per row with its attribution |

The paper also tested the camera-trap edit with a regional species list on [Felidae Conservation Fund](https://lila.science/datasets/felidae-conservation-fund/) images from LILA (CDLA-Permissive-1.0). That crop list is not in this repository.

Splits are by location, with no location shared between practice (photos used to fit a model), selection (photos used to choose between candidate models) and test. The camera-trap test uses 45 cameras that no practice crop comes from. The Galápagos natives in the island practice, selection and test sets come from three different islands: Isabela, San Cristóbal and Santa Cruz (iNaturalist places 49264, 29827 and 49265). [data/README.md](data/README.md) describes every file and field.

## Credits

Claude Fable 5.2, Claude Opus 5.5 and GPT-6 Astra Pro edited the weights for the general camera-trap edit, working in turn: Fable, then Opus, then GPT-6 Astra Pro, then Fable again. The Codex research assistant (OpenAI GPT) built and evaluated the island edit, the direct combined edit and the LoRA comparison.

## Citation and licence

Cite with [CITATION.cff](CITATION.cff):

> Andrew Hill (2026). *Can weight-space programming surface the domain expertise hidden in a small vision-language model? Evidence from camera traps and Galápagos species.* https://andrewxhill.com/frontier-model-weight-edits/paper.pdf

The code is licensed under [Apache-2.0](LICENSE). The camera-trap edit is Apache-2.0, as is its base model. The island edit, the LoRA model and the direct combined edit are CC BY-NC 4.0, because they were fitted on iNaturalist photos and iNaturalist's terms forbid using its data to train AI for commercial purposes. Data manifests keep their sources' terms. [LICENSES.md](LICENSES.md) lists every upstream licence and what it requires.
