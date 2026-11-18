# How the camera-trap edit was constructed

Rebuilt from public practice inputs, the camera-trap edit scores **81.76% camera-balanced accuracy**, a gain of **17.67 percentage points** over the unedited model's 64.09%, on the same 1,611 test crops. Its answers, raw replies and token sequences match the released edit's exactly. Three tensors differ by tiny amounts, so the rebuilt and released weights have different state hashes.

The editing models saved 39 numbered checkpoints of their work. The released camera-trap edit is the 39th, which the paper calls check 39. Checkpoints 36 to 38, named below, came just before it. Names such as `TL168` are capture banks in `recipe/selections047.json`, and the numbers in recipe file names are fixed names the builder expects.

The edit has two parts: a rewritten projector, the small network that passes image features to the language model, and fifteen changed rows of the output head. In this model the output head and the input word embeddings share one matrix, so each changed row changes both. The numerical programs used the labelled practice crops available to the editors, the model's answers on them, and vision and language activations captured while the model ran. The practice files exclude the 45 test cameras and the 204 tuning crops held back from the editors, which are not published.

## Projector and output rows

The projector writes animal-name hints into two image-token positions, 568 and 569. Each position has a branch covering the 58 animal names in `recipe/reference-metadata047.json`. Each branch writes one of two token vectors chosen by the recipe for that animal name. `class_tokens` in `recipe/math048.py` applies the name substitutions and token choices; a name that is a single token uses the same vector in both branches.

The recipe fits maps from the image features at those positions to SigLIP image and text features, and from those to a score for each animal name. It writes the resulting name-token vectors through selected projector units. Other fitted vectors separate the two positions, and separate the whole-image view from image tiles; subtracting their contribution silences existing units at position 569. `build_n3` in `recipe/math048.py` assembles these weights. Its `m8vec` correction subtracts the mean projector output at position 568, measured on practice crops. The final maps use all the practice feature banks combined.

A second correction uses the language model's final hidden activations to fit a residual direction for each option letter, A to K. `fit_readout_b` in `recipe/math048.py` uses examples with at least two options. For each letter it fits how far the probability of the correct option falls short, from the final hidden activations; its diagonal ridge term is `0.1 * trace(H.T @ diag(w) @ H) / 896`. Here `H` holds the hidden activations and `w` marks the examples that include that letter as an option. The final fit adds the `TL168` practice captures taken under checkpoint 38. [The bank table](CONSTRUCTION-SNIPPETS.md) names all five inputs.

The final head calculation takes the intermediate corrected head in float64 and adds four times each fitted residual direction to rows 32 to 42, the tokens for A to K. Row 362 is the token for a space followed by A; row 32 is A with no space. Row 362 is set to row 32 of the intermediate head plus 1.5 times the fitted space-A offset. The builder calls that intermediate head `E64`; it already holds the earlier format corrections. These twelve rows are cast to float32 and replaced. Earlier changes to rows 576, 785 and 151645 remain, giving fifteen changed rows in total.

[The arithmetic example](CONSTRUCTION-SNIPPETS.md) shows that last step. The executable recipe includes the maps, reference vectors and feature-bank calculations that come before it.

## Recompute from practice inputs

`build_edit.py` recomputes the feature banks, the reference text and image vectors and the fixed fits from the original public weights and the published practice inputs. It reads neither the released delta nor the released edited model, and no saved arrays from the original edit. The functions, the order they run in and the fixed selections are under `recipe/`; [RUN.md](RUN.md) gives the commands. This route recomputes the final recipe, not the editors' search for it.

The standalone builder ran in 80.94 minutes on Apple silicon (MPS), and its output reloaded through `from_pretrained`. Of 716 tensors, 713 are byte-identical to the released camera-trap edit and three differ. Both models map the same 717 tensor names onto those 716 tensors; the embedding and the head are two names for one tensor.

| State | SHA256 |
|---|---|
| Rebuilt from practice inputs | `c5e548f73914ec42134b409d9442339e5473be014057ce0d9ebe0a9636f58268` |
| Released camera-trap edit | `846cdbb58f3ca04c9ccda139d6e2c412921519f11da6e72a9a268480800b9881` |

| Tensor | Different values | Maximum absolute difference | Mean absolute difference (whole tensor) | Relative L2 |
|---|---:|---:|---:|---:|
| lm_head.weight | 9776 | 5.19678e-07 | 3.1301901e-12 | 3.4712419e-08 |
| model.multi_modal_projector.linear_1.bias | 112 | 0.00012207031 | 6.3010624e-06 | 3.4580425e-07 |
| model.multi_modal_projector.linear_1.weight | 126437 | 1.0058284e-06 | 1.469695e-08 | 7.035199e-07 |

The head differences are confined to rows 32 to 42, which are also the shared embedding rows. The 713 matching tensors include the projector's `linear_2` matrix. The cause of the three differences is not known.

## Test of the rebuilt model

The test took 22.51 minutes. The rebuilt model scored 81.76% camera-balanced accuracy, the same as the released edit, and 83.86% plain accuracy, with no missing answers. All 1,611 raw replies, token sequences, parsed answers and faults match the released edit's. The gain over the unedited model was 17.67 percentage points, with a 95% interval of 12.41 to 23.61 from resampling whole cameras. 349 crops the unedited model got wrong became correct, and 33 it got right became wrong.

The rebuilt `tokenizer_config` stores `padding_side=left`; the released file stores `right`. The vocabulary, chat template, processor and generation settings otherwise match. The evaluator sets left padding for both models and uses batch one. The rebuilt model was scored with its own saved files.

To reproduce the released weights exactly, use `apply_edit.py` and the delta.
