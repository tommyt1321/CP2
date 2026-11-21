# CP2: A3 retinal field-of-view pilot (v1)

This pilot compares three ways of defining image regions for the same frozen A3 classifier. It uses the original 527-image APTOS validation set and the existing 384×384 RGB PNGs, Python environment and model weights. It does not train a model, read APTOS test or IDRiD images, or change the primary model, evaluations, Figma design or Gradio interface. The field-of-view (FOV) mask estimates the outer edge of the visible fundus; it does not label lesions.

## Run in Windows Git Bash

If installing from the pilot ZIP, extract it at the CP2 root. The archive contains only `experiments/aptos_fov_v1/`. The example assumes it is in Downloads. `unzip -n` skips existing files. Keep the directory structure; do not move the scripts to the project root.

```bash
cd /c/CP2
unzip -n ~/Downloads/CP2_A3_FOV_Pilot_v1.zip
python -B experiments/aptos_fov_v1/run_fov.py preview
```

1. **Preview first.** Inspect all 527 validation images using RGB alone; no model weights are loaded. Compare the teal outline in `sample_01.png` and every `flagged_*.png` with the original RGB. Check whether the mask cuts off dark retinal tissue or includes the black camera border. `mask_audit.csv` records each image; `masks/<image_id>.png` contains its binary mask. An unflagged image has not necessarily passed visual review. A blank image or one with no identifiable fundus is recorded as a failure; the script does not substitute the full canvas.
2. **Smoke test.** Run all three branches on three validation images selected in advance by hashing their IDs. Use the frozen A3 weights, check the original prediction for each image against the validation predictions saved during training, and generate top-10% comparison panels. This step does not report accuracy.
3. **Full comparison.** Use the same weights for paired predictions on all 527 images. Review the preview and smoke test first. If a mask cuts off retinal tissue, keep that record; do not drop the image before reporting scores.

```bash
python -B experiments/aptos_fov_v1/run_fov.py smoke
python -B experiments/aptos_fov_v1/run_fov.py compare
```

Each invocation creates `experiments/aptos_fov_v1/runs/<UTC_timestamp>_<mode>_<random_id>/`. There are no options to overwrite, resume or choose an arbitrary output path. If a run fails, keep the error and the per-image records already written. A retry creates a new directory. The experiment's `.gitignore` excludes run outputs.

Use the original CP2 environment (including Python, NumPy, Pillow, SciPy, PyTorch, torchvision and PyG). Package versions, upstream code and data manifests must match the training records. No new weights are needed. If SciPy is missing, record the error before changing the working environment. Use `--device cpu` to select the CPU explicitly. Only the model uses the GPU; region construction runs on the CPU.

## What the three branches compare

| Branch | Region definition | Question |
|---|---|---|
| `original_A3` | Reuse the original boundary-aware SLIC partition and A3 forward pass on the full canvas, including the black border. | Can the baseline predictions be reproduced on the same validation images? |
| `original_regions_clipped` | Start from the same partition, remove pixels and nodes outside the estimated FOV, and count disconnected fragments as separate regions. | How do predictions change when the original regions cover only the fundus and their adjacencies change? |
| `FOV_resegmented` | Place 256 initial seeds inside the estimated FOV. Assignment, updates and connectivity merging exclude outside pixels from region membership. | How do predictions and region rankings change when the region budget is concentrated inside the fundus? |

Both FOV branches keep the same RGB input and DenseNet feature extraction. Region pooling, graph adjacency, GAT, ViT and Rollout use only valid regions, with no background nodes. Area is still normalized by 384², and centroid coordinates use the original coordinate system. DenseNet's receptive fields can still see the border. Resegmentation also changes seed positions, spatial step size and boundary-strength normalization inside the FOV, so its effects cannot be attributed solely to excluding the black border. The target is 256 seeds; the final region count may change after connectivity processing.

## Read the outputs

- **Preview:** `summary.json`, `mask_audit.csv`, representative thumbnails and all flagged images.
- **Smoke test and full comparison:** `paired_predictions.csv` records the ground-truth grade, three predictions and their five-class probabilities, actual region counts, and the numbers of selected pixels overall and outside the estimated FOV. Teal marks selected pixels inside the FOV; pink marks selected pixels outside. Neither colour denotes a DR lesion. The top 10% is a fraction of the actual number of regions, rounded up; the selected pixel area need not be 10%.
- **Full comparison:** `summary.json` gives QWK, macro-F1, accuracy and confusion matrices for each branch, differences from the original, and counts of changed predictions. Per-image mask warnings are retained. Metrics are written only when all 527 images have paired results and all original predictions match the archived A3 epoch-21 predictions by image ID. The script also checks the A3 weight hash and upstream files it reads. A `REVIEW_MASKS` flag calls for visual inspection before interpreting the result.
- Zero selected pixels outside the estimated FOV is guaranteed by construction in the two experimental branches. It does not establish better grading or lesion localization. The validation set was used for model selection; this is a development comparison, not a new independent test. The primary IDRiD results remain unchanged.

## Method sources

- Papadopoulos, Topouzis & Delopoulos (2021), [An interpretable multiple-instance approach for the detection of referable diabetic retinopathy in fundus images](https://doi.org/10.1038/s41598-021-93632-8). This motivated filtering out-of-FOV patches before classification. This pilot does not use their 50% patch threshold.
- Irving (2016), [maskSLIC](https://arxiv.org/abs/1606.09518). This motivates generating superpixels inside a mask. The CP2 adaptation uses the project's boundary detector, grid paths, connected components and trained neural modules; it is not the paper's official implementation.
