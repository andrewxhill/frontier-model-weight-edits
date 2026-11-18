# Results

`summary.json` holds every headline score. `python code/summarise_results.py --output results/summary.json` recomputes it from the per-crop files below.

| Folder | Contents |
|---|---|
| `camera/` | One answer per test crop on the 1,611-crop camera benchmark, for the unedited model, the four released models and SigLIP alone (each crop's image features matched to the text features of each name on its list) |
| `island/` | One answer per crop on the 697-crop island test, for the same models plus SpeciesNet's standard output and its raw top label |
| `check39-detail/` | Camera-trap edit only: answers when no list is given, scores with each part of the edit removed, scores for every checkpoint the editors saved, and the regional species-list test on Felidae Conservation Fund images |

File names follow the Hugging Face repositories: `camera-trap-edit` is the camera-trap edit (check 39 in the paper, the 39th checkpoint the editing models saved), `galapagos-island-edit` the island edit, `camera-island-lora` the LoRA model and `camera-island-direct-edit` the direct combined edit.

In `island/`, `native_status_correct` is the decision the native-versus-look-alike scores use: a land or marine iguana counts as native when the answer names either native iguana genus. `genus_correct` gives rat recall. SpeciesNet and SigLIP answers were mapped to names with SpeciesNet's published taxonomy, and their files hold those saved decisions. `code/island/score.py` rescores the vision-language model files from their `answer` field.
