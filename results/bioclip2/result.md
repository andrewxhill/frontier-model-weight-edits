# BioCLIP 2 on the same camera and island crops

18 November 2026.

BioCLIP 2 with taxonomic-plus-common text reaches **84.07% balanced accuracy** on the camera bench, beside 81.76% for the AI weight edit, check 39, and 64.09% for unedited LLaVA-OneVision 0.5B. On the island photos, its taxonomic-plus-common prompts score 94.49% for land-iguana versus green-iguana status, 94.48% for marine-iguana status and 100.00% for mockingbird versus other-bird status. All rows reuse the existing crops and scorers.

BioCLIP 2’s published training catalogue contains **399 of the 684 iNaturalist test photo IDs**. The island comparison therefore includes photos present in its training data. The URL check recognises standard iNaturalist image links and supported regional S3 links; unusual URL forms, alternate-provider copies and near-duplicates remain unchecked. The 285 unmatched IDs are not proof of unseen images. The 13 camera-trap crops were outside this iNaturalist check.

## Camera bench

All rows use the same 1,611 crops and 45 cameras. Balanced accuracy averages class recalls within each camera, then gives each camera equal weight. Plain accuracy counts crops equally. Each camera’s option list was built from its labels in the original study; this is an assisted naming task.

The edits and LoRA all modify LLaVA-OneVision 0.5B. Check 39 is the delivered AI weight edit; Island edit is the separate island specialist. Combined direct edit was built to combine their behaviour. LoRA (low-rank adaptation) is the gradient-trained comparison starting from check 39. SigLIP is the zero-shot reference from the original vision encoder. The paper gives their construction and training details.

BioCLIP chooses the menu entry with the highest image–text similarity. Taxonomic + common includes the scientific lineage and common name; scientific + common uses the genus/species name and common name. Both retain common-name-only text for 21 coarse camera labels.

| Arm | Balanced % | Plain % | Correct / 1,611 | Non-answers |
| --- | --- | --- | --- | --- |
| Unedited LLaVA | 64.09 | 64.25 | 1035 | 237 |
| Check 39 | 81.76 | 83.86 | 1351 | 0 |
| Island edit | 72.14 | 74.80 | 1205 | 3 |
| Combined direct edit | 80.70 | 83.36 | 1343 | 0 |
| LoRA | 79.22 | 83.24 | 1341 | 0 |
| SigLIP zero-shot | 82.79 | 86.28 | 1390 | 0 |
| BioCLIP 2 taxonomic + common | 84.07 | 85.85 | 1383 | 0 |
| BioCLIP 2 scientific + common | 84.36 | 86.34 | 1391 | 0 |

| BioCLIP row | Compared with | Balanced-accuracy gain, percentage points | 95% interval for gain | Fixes | Harms |
| --- | --- | --- | --- | --- | --- |
| BioCLIP 2 taxonomic + common | Unedited LLaVA | +19.98 | +13.76 to +26.75 | 448 | 100 |
| BioCLIP 2 taxonomic + common | Check 39 | +2.31 | -1.58 to +6.38 | 174 | 142 |
| BioCLIP 2 scientific + common | Unedited LLaVA | +20.27 | +14.01 to +26.86 | 447 | 91 |
| BioCLIP 2 scientific + common | Check 39 | +2.60 | -1.19 to +6.54 | 172 | 132 |

Fixes count crops the comparison model got wrong and BioCLIP got right; harms count the reverse. Intervals for the balanced-accuracy gain use 2,000 paired whole-camera draws, seed 20261112. They describe camera variation conditional on these fixed methods and reused tests. SpeciesNet has no camera-bench row in this comparison.

## Island native versus look-alike decisions

The 697 accepted crops comprise 684 iNaturalist photographs and 13 camera-trap cat crops. Native taxa come from Galápagos photographs; look-alikes and controls also include worldwide photographs. There were 752 eligible records; 19 fetch failures and 36 records without an accepted animal crop remain excluded under the original rule. Every new BioCLIP row uses the same accepted pixels and three fixed 11-name menus.

The status score groups the native land and marine iguana genera together; the genus score requires the appropriate land or marine genus. Mockingbirds must be identified as Mimus, their genus. On other-bird photos, status accepts a bird classification that excludes Mimus, including a suitable broader family or order; an unidentified bird alone is insufficient.

SpeciesNet stock is its country-filtered standard answer; raw is the unfiltered classifier choice. Those two rows retain unrestricted outputs and received no menus. LLaVA, SigLIP and BioCLIP use the fixed menus.

| Arm | Land status % | Land genus % | Marine status % | Marine genus % | Mockingbird status % | Mockingbird genus % | Rattus recall % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Unedited LLaVA | 50.00 | 49.17 | 50.00 | 0.85 | 70.83 | 49.17 | 2.96 |
| Check 39 | 51.83 | 51.00 | 51.83 | 2.68 | 91.67 | 45.00 | 13.75 |
| Island edit | 81.22 | 78.72 | 80.37 | 69.36 | 98.33 | 95.00 | 81.23 |
| Combined direct edit | 55.49 | 54.65 | 55.49 | 7.18 | 90.83 | 44.17 | 17.70 |
| LoRA | 87.58 | 83.41 | 91.69 | 84.06 | 93.33 | 93.33 | 88.20 |
| SigLIP zero-shot | 78.29 | 73.29 | 77.45 | 73.21 | 98.33 | 83.33 | 46.86 |
| SpeciesNet stock | 12.44 | 12.44 | 12.44 | 12.44 | 3.33 | 0.00 | 8.11 |
| SpeciesNet raw | 27.32 | 27.32 | 27.32 | 27.32 | 17.50 | 0.83 | 12.76 |
| BioCLIP 2 taxonomic + common | 94.49 | 93.66 | 94.48 | 91.94 | 100.00 | 96.67 | 83.29 |
| BioCLIP 2 scientific + common | 94.49 | 93.66 | 94.48 | 90.24 | 100.00 | 96.67 | 86.32 |

Each comparison has two sides: the native animal and its look-alike. On each side, correctness is averaged within each public location grouping, then those groupings receive equal weight. The two sides then receive equal weight. Rattus recall gives each of the 76 public location groupings equal weight across 112 rat crops.

| Comparison | Side | Crops | Public location groupings |
| --- | --- | --- | --- |
| Land iguana versus green iguana | Green iguana | 57 | 41 |
| Land iguana versus green iguana | Native land iguana | 60 | 1 |
| Marine iguana versus green iguana | Green iguana | 57 | 41 |
| Marine iguana versus green iguana | Native marine iguana | 59 | 1 |
| Mockingbird versus other birds | Native mockingbird | 60 | 1 |
| Mockingbird versus other birds | Other birds | 60 | 1 |

Native rice rats have zero accepted crops, so native rice-rat versus Rattus balanced accuracy is unavailable. The other-bird group consists of 60 brown-noddy photographs. No accepted crops represent Polynesian rats, Galápagos petrels, Galápagos doves, yellow warblers or blue-footed boobies. Each native group has one coarse public county, which is not a verified independent field site. These public location groupings do not support population-level uncertainty estimates for native taxa or pooled rat recall.

## Exact species

Each cell is equal-public-location exact-species recall (%); n counts accepted crops. Base means unedited LLaVA; Island is the island specialist; Combined is the direct combined edit. Bio tax and Bio sci are the two frozen BioCLIP text formats. Full domain, query, genus, primary, species, non-answer and paired tables are in [the numerical record](summary.json).

| Domain | Common name | Reference species | n | Base | C39 | Island | Combined | LoRA | SigLIP | SN stock | SN raw | Bio tax | Bio sci |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| camera_trap | domestic cat | Felis catus | 13 | 8.33 | 47.62 | 100.00 | 63.10 | 33.33 | 25.00 | 48.81 | 70.24 | 25.00 | 33.33 |
| citizen_photo | marine iguana | Amblyrhynchus cristatus | 59 | 1.69 | 1.69 | 76.27 | 3.39 | 77.97 | 89.83 | 0.00 | 0.00 | 93.22 | 89.83 |
| citizen_photo | brown noddy | Anous stolidus | 60 | 0.00 | 0.00 | 93.33 | 1.67 | 86.67 | 66.67 | 0.00 | 0.00 | 93.33 | 93.33 |
| citizen_photo | domestic goat | Capra hircus | 46 | 63.75 | 66.25 | 72.50 | 68.75 | 95.00 | 73.75 | 31.25 | 41.25 | 83.75 | 87.50 |
| citizen_photo | Santa Fe land iguana | Conolophus pallidus | 11 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 45.45 | 0.00 | 0.00 | 90.91 | 90.91 |
| citizen_photo | Galapagos land iguana | Conolophus subcristatus | 49 | 91.84 | 95.92 | 95.92 | 95.92 | 73.47 | 79.59 | 0.00 | 0.00 | 79.59 | 79.59 |
| citizen_photo | domestic cat | Felis catus | 29 | 84.62 | 92.31 | 96.15 | 92.31 | 92.31 | 84.62 | 75.00 | 86.54 | 88.46 | 88.46 |
| citizen_photo | common green iguana | Iguana iguana | 57 | 0.00 | 3.66 | 62.44 | 10.98 | 90.16 | 56.59 | 24.88 | 54.63 | 90.65 | 90.65 |
| citizen_photo | Galapagos mockingbird | Mimus parvulus | 60 | 40.00 | 75.00 | 96.67 | 73.33 | 100.00 | 11.67 | 0.00 | 0.00 | 63.33 | 70.00 |
| citizen_photo | house mouse | Mus musculus | 55 | 0.00 | 6.37 | 55.69 | 6.37 | 51.49 | 61.11 | 0.00 | 0.00 | 52.30 | 55.96 |
| citizen_photo | South American coati | Nasua nasua | 59 | 0.88 | 13.16 | 97.37 | 0.88 | 92.98 | 89.04 | 72.37 | 84.12 | 94.74 | 93.42 |
| citizen_photo | European rabbit | Oryctolagus cuniculus | 49 | 0.00 | 0.00 | 79.49 | 8.01 | 92.31 | 73.08 | 17.31 | 36.54 | 89.74 | 87.18 |
| citizen_photo | brown rat | Rattus norvegicus | 55 | 5.92 | 25.39 | 70.44 | 30.66 | 34.39 | 48.64 | 0.00 | 3.51 | 82.54 | 83.07 |
| citizen_photo | black rat | Rattus rattus | 57 | 0.00 | 0.00 | 31.00 | 0.00 | 76.17 | 5.83 | 0.00 | 0.00 | 63.50 | 66.50 |
| citizen_photo | pig | Sus scrofa | 38 | 23.04 | 36.76 | 37.25 | 36.76 | 66.18 | 72.55 | 60.29 | 63.24 | 79.41 | 69.12 |

## How BioCLIP was run

The official MIT-licensed `imageomics/bioclip-2` checkpoint is pinned to `2957b322090f9cb17ae72c71981c7218a28d81e0`; its weight-file SHA-256 is `b7b2bf6fbc95799e42630e394cf95803892ab447c1a8ab629dbc82fbeaf7dfef`. The image encoder and text encoder produce vectors; BioCLIP selects the menu name with the highest similarity to the image vector. The maker’s biological image head ran in float32 on a Mac Studio using Apple’s Metal Performance Shaders (MPS), batch one, with its validation resize/centre-crop and 81-template text ensemble. Each template vector was normalised, averaged and normalised again. Both documented formats were fixed before scores; neither was chosen by its result.

Scientific taxa were mapped from menu entries using public GBIF exact-name receipts. Twenty-one coarse camera labels, such as bird and rat, keep common-only prompts in both variants; their wording stays at the coarse level. Five iguana taxa use GBIF’s Squamata class with an empty order slot. The strings use current GBIF mappings, which can differ from publication-time training labels.

Camera RGB inputs use the existing 25% padded dataset boxes; island RGB inputs use the existing tight MegaDetector crops. Hashes verified the same pixels before each model’s own preprocessing. The earlier BioCLIP apparatus check reproduced 9,202/11,983 Rare Species answers, 76.7921%, against 76.8% published, using the original scientific-plus-common recipe.

SpeciesNet stock and raw retain their original outputs and did not receive the supplied menus. Its taxonomy contains Mimus, though not the Galápagos mockingbird species. The camera and island tests have already been read in this study, and pretraining exposure is possible for all models. These rows describe the same crops; they do not establish unseen-image generalisation or human replacement.

## Records

The original BioCLIP training-catalogue scan processed all 213,937,319 rows of revision `a8f38b4388579862c56ae57d6f094c2ac0e92e12`. Its 399 matched source photo IDs link to 399 test observations and 503 catalogue rows. The independent check verified saved scan outputs and did not repeat the catalogue scan.

An independent calculation agrees on all 2,308 new raw vector records and 173,531 checked values, including every existing comparison row. It uses separately written code and the same frozen scoring definitions.

Primary sources: Gu et al. (2025), [BioCLIP 2](https://arxiv.org/abs/2505.23883); the [official model card](https://huggingface.co/imageomics/bioclip-2); and the [original TreeOfLife catalogue](https://huggingface.co/datasets/imageomics/TreeOfLife-200M/tree/a8f38b4388579862c56ae57d6f094c2ac0e92e12). Research procedures used Kassis, Agarwal, He, Patel and Brueckner (2026), [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065).
