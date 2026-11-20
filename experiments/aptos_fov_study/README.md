# APTOS field-of-view sensitivity analysis (frozen A3)

This is a supplementary validation experiment. The full-canvas A3 remains the primary model for the prototype. Do not use the FOV branches to replace the main APTOS/IDRiD results. No retraining, APTOS test images or IDRiD images were used for the comparison.

## Files and workflow

Run commands from the CP2 root, using the existing CP2 environment. The v1 directory must keep its original name and location: both pilot modules are imported and hash-checked by the v3 runner. The four later scripts can live in this directory even if the original run used copies in Downloads. The v1 README documents the **pilot**, not the final v3 masks. You do not need to rerun inference simply to archive the experiment.

1. Existing upstream dependencies: locked validation split and preprocessed RGBs, archived A3 validation predictions, selected epoch-21 checkpoint, pinned CP2 code and environment.
2. Pilot: see ../aptos_fov_v1/README.md. The v2 geometry step uses its fov_regions.py.
3. Generate v2 candidate masks: python -B experiments/aptos_fov_study/aptos_fov_corner_preview_v2.py
4. Export v2 visual review: python -B experiments/aptos_fov_study/collect_aptos_fov_v2_review.py (review helper, not a model dependency). Preserve the external review and frozen decisions; do not select images based on classification outcomes.
5. Generate v3 geometry from the matched v2 run: python -B experiments/aptos_fov_study/aptos_fov_geometry_preview_v3.py
6. With the originally accepted v3 run in place, the paired comparison is python -B experiments/aptos_fov_study/aptos_fov_a3_compare_v3.py compare. The runner matches recorded protocol, run summary, audit and masks. Preserve the original run folders; a new geometry run need not satisfy the identity checks on the archived run. Do not change hash checks.

## Recorded result (527 locked APTOS validation images)

| Region construction | QWK | Macro-F1 | Correct / 527 | Accuracy |
|---|---:|---:|---:|---:|
| Original A3, full canvas | 0.8885 | 0.6767 | 434 | 82.35% |
| New partition inside the estimated FOV | 0.8955 | 0.6223 | 426 | 80.83% |
| Clip original regions to the estimated FOV | 0.8601 | 0.6152 | 415 | 78.75% |

The primary comparison was original versus FOV resegmentation. QWK increased slightly; accuracy and macro-F1 decreased. For true grade 1, correct predictions fell from 30/51 to 11/51. The same full RGB was supplied to DenseNet in all arms. Zero selected pixels outside the estimated field in the FOV arms is enforced by construction and does not establish better lesion localisation or isolate a causal effect of border pixels.

## Evidence retention (outside Git)

Keep the completed runs below in their original locations and back them up to another private drive in the same relative folder structure:

- experiments/aptos_fov_corner_review_v2/runs/ — the matched run including summary.json, mask_audit_v2.csv and all 527 candidate_masks PNGs; v3 geometry reads them.
- experiments/aptos_fov_geometry_v3/runs/20261120_073956_77ed8cfd9a/ — protocol.json, summary.json, per_image.csv, all 527 candidate_masks PNGs and the review sheets. The v3 geometry ZIP does NOT contain the full masks.
- experiments/aptos_fov_a3_compare_v3/runs/ — the completed original compare run including protocol.json, summary.json, paired_predictions.csv and its six PNG previews; retain any technical-gate record. The compare ZIP contains its 527 prediction rows, not the input masks.
- experiments/aptos_fov_v1/runs/ — keep original preview/smoke/compare evidence if run. Also keep the existing frozen dataset, checkpoint, environment and upstream audit; they do not belong in this commit.

Keep these exported packets in the backup (outside Git):

- CP2_FOV_v2_visual_review_20261120_071538_5d5394b1.zip — SHA-256: cf7201726d9ea574b068854b17b5140a0df1e6af82273db94593bf6e9c728fb6
- CP2_FOV_v3_geometry_20261120_073956_77ed8cfd9a.zip — SHA-256: 8c7cbe2f521b8256e156708a0ce2cc67b6cbfaf437495be196b9a31503098ad2
- CP2_A3_FOV_v3_20261120_111127_compare_1e0068a5da.zip — SHA-256: d202e45367cb8a3b62dd02f19436e89072e03d27f7f2601897b6545c705d9d90
- APTOS_FOV_native_review_evidence.zip — SHA-256: 65b11212b5b108b16ffb98d657162d6637aed6acc7c84ab751bddf4ca83a894b

The results-review HTML is optional for the report/defence. The IDRiD_61 display toggle is a separate UI example, not an APTOS validation result. Remove only caches and byte-identical extra downloads after verifying an independent backup. Do not delete whole runs or the only copy of any evidence.
