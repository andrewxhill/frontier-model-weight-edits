# How the edit was constructed

The practice-input build reproduced **81.76% camera-balanced accuracy**, a gain of **17.67 percentage points** over the unedited model's 64.09%, on the same 1,611 crops. Its answers, raw replies and token sequences matched released check 39. Three tensors differ numerically, so the practice build and released weights have different state identities.

The edit combines a rewritten multimodal projector, which maps image features into the language model, with fifteen changed rows of the tied language embedding/output head. The numerical programs use the labelled practice crops available to the editors, model predictions and captured vision and language activations. The practice manifests exclude private tuning crops and the 45 test cameras.

## Projector and output rows

The projector inserts animal-name hints into two image-token positions, 568 and 569. Each position has a branch covering the 58 animal names in `recipe/reference-metadata047.json`. Each branch writes one of two token vectors selected by the recipe for that animal name. `class_tokens` in `recipe/math048.py` applies the source name substitutions and token choices; a one-token form uses the same vector in both branches.

The recipe fits maps from image features at those positions to SigLIP image/text features and then to class scores. It writes the resulting name-token vectors through selected projector units. Other fitted vectors distinguish the two positions and the whole-image view from image tiles; subtracting their contribution suppresses existing units at position 569. `build_n3` in `recipe/math048.py` assembles these weights. Its `m8vec` correction subtracts a captured mean projector output at position 568. The final maps use the combined practice feature banks.

A second correction uses captured language hidden states to fit residual directions for option letters A–K. `fit_readout_b` in `recipe/math048.py` uses examples with at least two options. For each letter it fits the correct-option probability residual from final hidden activations; its diagonal ridge term is `0.1 * trace(H.T @ diag(w) @ H) / 896`. Here `H` contains the hidden activations and `w` marks examples that include that letter as an option. The final fit includes the `TL168` practice captures from the preceding check-38 candidate. [The bank table](CONSTRUCTION-SNIPPETS.md) names all five inputs.

The final head calculation takes the intermediate corrected head in float64 and adds four times each fitted residual direction to rows 32–42. Row 362 is the token for a space followed by A; row 32 is the token A without a space. Row 362 becomes row 32 from the already format-corrected intermediate head, plus 1.5 times its fitted offset. The builder names that intermediate head `E64`; it precedes the final letter-residual additions. These twelve rows are cast to float32 and replaced. Earlier changes to rows 576, 785 and 151645 remain, giving fifteen changed rows in total. The input embeddings and output head share the same values.

[The arithmetic example](CONSTRUCTION-SNIPPETS.md) shows that last step. The executable recipe includes the preceding maps, reference vectors and feature-bank calculations.

## Recompute from practice inputs

`build_edit.py` recomputes the feature banks, reference text/image vectors and fixed fits from original public weights and the disclosed practice inputs. It reads neither the final delta nor the released edited checkpoint or old numerical arrays. The source functions, dependency order and fixed selections are under `recipe/`; [RUN.md](RUN.md) gives the commands. This route recomputes the final recipe rather than the editors' search for it.

One execution of the standalone builder on Apple silicon MPS completed in 80.94 minutes and reloaded through `from_pretrained`. Its full base-anchored state identity is `c5e548f73914ec42134b409d9442339e5473be014057ce0d9ebe0a9636f58268`. Of 716 canonical tensors, 713 are byte-identical to the released `846c` state and three differ. Both models have the same valid mapping of 717 names to canonical tensors and tied aliases.

| Tensor | Different values | Maximum absolute difference | Mean absolute difference (whole tensor) | Relative L2 |
|---|---:|---:|---:|---:|
| lm_head.weight | 9776 | 5.19678e-07 | 3.1301901e-12 | 3.4712419e-08 |
| model.multi_modal_projector.linear_1.bias | 112 | 0.00012207031 | 6.3010624e-06 | 3.4580425e-07 |
| model.multi_modal_projector.linear_1.weight | 126437 | 1.0058284e-06 | 1.469695e-08 | 7.035199e-07 |

The head differences are confined to rows 32–42, which are also the tied embedding rows. The 713 matching tensors include the projector's `linear_2` matrix. The cause of the three tensor differences remains unresolved.

## Test of the rebuilt state

The fixed test took 22.51 minutes. It gave 81.76% camera-balanced accuracy, beside the released check-39 score of 81.76%, and 83.86% plain accuracy, with zero missing answers. A separate checker confirmed that all 1,611 raw replies, token sequences, parsed answers and faults matched check 39. The gain over unedited was 17.67 percentage points, with a whole-camera 95% interval of 12.41–23.61. There were 349 previously wrong crop answers that became correct and 33 previously correct answers that became wrong.

The rebuilt `tokenizer_config` stores `padding_side=left`; the released file stores `right`. Vocabulary, chat, processor and generation assets otherwise match. The evaluator explicitly sets left padding for both models and uses batch one. The rebuilt state was scored with its own saved assets.

Use `apply_edit.py` and the delta to reconstruct the released `846c` state exactly. Tensor errors, predictions and verification records are in `results/source-construction/`.
