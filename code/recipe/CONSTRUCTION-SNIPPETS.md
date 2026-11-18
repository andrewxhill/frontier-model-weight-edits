# Final output-row calculation

The camera-trap edit gained 17.67 percentage points on the camera-trap test with each camera's list of names. This example shows the last calculation in that edit, which changes the output rows for the answer letters. `build_edit.py` runs the whole recipe from practice inputs; `apply_edit.py` replays the released delta exactly.

The calculation takes the intermediate head, which already holds earlier corrections, and the fitted coefficients:

```python
E64 = intermediate_corrected_head.astype(float64)
E2[32:43] = E64[32:43] + 4.0 * R_all2
E2[362] = E64[32] + 1.5 * alias_direction
replacement_rows = E2[list(range(32, 43)) + [362]].astype(float32)
```

`E2` is the new head. Rows 32 to 42 are the letters A to K, and row 362 is a space followed by A. The intermediate head already holds the changes to rows 576, 785 and 151645, so these twelve rows bring the total to fifteen. The calculation needs that intermediate head, not the unedited base.

`R_all2` is an 11-by-896 matrix: one residual direction per letter, in the language model's 896-wide hidden space. `recipe/build_edit048.py`, which `build_edit.py` runs, builds it from these five banks of practice captures. The capture names in the right-hand column are keys in `recipe/selections047.json`, which fixes their crop IDs. The editors saved 39 numbered checkpoints, and the released edit is checkpoint 39; some banks were captured under earlier checkpoints, which the builder reconstructs.

| Bank in the builder | Practice captures |
|---|---|
| `BK_A` | `T1`, with the first format-corrected head |
| `BK_B` | `LATE21`, `LATE22` and selected `LATE23` and `LATE25` captures combined, with the intermediate last-MLP contribution removed where the builder specifies |
| `BK_C` | `TR320`, captured under checkpoint 37 |
| `BK_D` | `DEV256`, captured under checkpoint 36 |
| Final added bank | `TL168`, captured under checkpoint 38 |

The space-A offset comes from the earlier intermediate head. The projector also needs fitted maps, the original weights, reference vectors, the order of its units, vectors that tell positions and content apart, and a mean-correction vector measured on practice crops. [Construction details](CONSTRUCTION.md) explain how the full recipe recomputes these inputs and report its three small numerical differences from the released weights.
