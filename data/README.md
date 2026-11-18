# Data manifests

These files list every crop the study used: where its photo comes from, the box that cuts the crop, its label and its split. They contain no images. `code/eval/fetch_images.py` and `code/island/fetch_crops.py` download the photos from their public sources and check each file's SHA256 hash.

## Files

| File | Rows | What it is |
|---|---:|---|
| `camera/manifest.json`, `camera/crops.jsonl` | 1,611 | Camera-trap test: 45 cameras from Island Conservation and Caltech Camera Traps, with each camera's list of names |
| `camera/practice/crops.json`, `camera/practice/cameras.json` | 23,447 | Practice crops from 57 other cameras, used to build the camera-trap edit |
| `combined/camera-practice.jsonl` | 2,095 | Camera practice crops used to build both one-model-for-both-jobs models |
| `combined/camera-selection.jsonl` | 194 | Camera crops used to choose among the direct combined edit's eight candidates |
| `island/practice.jsonl` | 895 | Island practice photos; `used_in` marks the 846 fitted by the island edit and the 807 used by both combined models |
| `island/selection.jsonl` | 456 | Island photos used to choose among candidate edits; 426 gave a usable crop |
| `island/test.jsonl` | 752 | Island test photos; 697 gave a usable crop and form the test |

Rows with `crop_status` other than `complete` were sampled but gave no crop: the download failed or the detector found no animal. They are kept so the sample is complete.

## Splits

Practice photos fit a model, selection photos choose between candidate models, and test photos are scored once. Splits are by location: no camera, island or other location appears in more than one of the three.

- **Camera traps.** The 45 test cameras supply no practice or selection crop.
- **Galápagos natives.** Practice photos come from Isabela (iNaturalist place 49264), selection from San Cristóbal (29827) and test from Santa Cruz (49265). Each native group therefore has one place per split.
- **Look-alikes and controls** (green iguanas, rats, other birds, Island Conservation camera-trap frames). These come from many locations worldwide, each assigned to one split.

## Island fields

| Field | Meaning |
|---|---|
| `id` | `inat:<observation id>` or `ic:<Island Conservation image>` |
| `source`, `url`, `observation_url`, `photo_id` | Where the photo comes from |
| `licence`, `attribution` | The photo's own licence and credit line, as published by its source |
| `source_image_sha256` | SHA256 of the downloaded file |
| `integer_box` | Crop box in pixels, `[x0, y0, x1, y1]`, from MegaDetector v5a |
| `pixel_sha256` | SHA256 of the crop's RGB pixels; also the crop's file name |
| `task` | Which 11-name list the crop is shown with: `iguana`, `bird` or `rodent` |
| `group` | Scoring group, such as `native_land_iguana`, `green_iguana`, `native_mockingbird`, `other_birds` or `invasive_rattus` |
| `truth_species`, `truth_genus` | Reference label from the research-grade observation or the camera-trap annotation |
| `location` | Location used for equal weighting and for the split |

## Sources and terms

| Source | Licence | Notes |
|---|---|---|
| [Island Conservation Camera Traps](https://lila.science/datasets/island-conservation-camera-traps/) | CDLA-Permissive-1.0 | LILA withholds images that contain people. No row here has a human label. |
| [Caltech Camera Traps](https://lila.science/datasets/caltech-camera-traps/) | CDLA-Permissive-1.0 | Cite Beery, Van Horn and Perona, *Recognition in Terra Incognita*, ECCV 2018. |
| [iNaturalist](https://www.inaturalist.org/) | Per photo: CC0, CC-BY or CC-BY-SA | Credit each photo with its `attribution` and licence when you use it. iNaturalist's terms of use prohibit using its data to train AI models for commercial purposes. |

The CDLA-Permissive-1.0 text is at https://cdla.dev/permissive-1-0/. These manifests are published under the same terms as their sources.
