# Final output-row calculation

The full edit gained 17.67 percentage points on the with-list test. This example shows the last output-row calculation in that edit. The executable practice-input recipe is `build_edit.py`; exact delta replay is `apply_edit.py`.

The final output-row calculation takes the already-corrected intermediate head and fitted coefficients:

```python
E64 = intermediate_corrected_head.astype(float64)
E2[32:43] = E64[32:43] + 4.0 * R_all2
E2[362] = E64[32] + 1.5 * alias_direction
replacement_rows = E2[list(range(32, 43)) + [362]].astype(float32)
```

Here `E2` is the destination head. The intermediate head already contains corrections to rows 576, 785 and 151645. The final twelve-row operation brings the total to fifteen changed rows. It therefore requires that intermediate head, not the unedited base.

`R_all2` is an eleven-by-896 residual matrix. `build_edit048.py` constructs it from these five banks, whose crop IDs are fixed in `recipe/selections047.json`:

| Bank in the builder | Practice captures |
|---|---|
| `BK_A` | `T1`, with the initial format-corrected head |
| `BK_B` | Combined `LATE21`, `LATE22`, selected `LATE23` and selected `LATE25` captures, with the intermediate last-MLP contribution removed where specified in the builder |
| `BK_C` | `TR320`, captured under the reconstructed check-37 state |
| `BK_D` | `DEV256`, captured under the reconstructed check-36 state |
| Final added bank | `TL168`, captured under the reconstructed check-38 state |

The space-A offset is derived from the earlier intermediate head. The projector also requires fitted maps, original weights, reference/prototype vectors, unit order, position/content discriminants and a captured mean-correction vector. [Construction details](CONSTRUCTION.md) explain how the full recipe recomputes these inputs and report its three numerical differences from the released weights.
