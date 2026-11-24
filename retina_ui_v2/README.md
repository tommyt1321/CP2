# Retina Insight · local Gradio workbench

This is the implementation of the approved Figma interface for the CP2 project. Live analysis uses the frozen A3 checkpoint from the local project. It does not replay Figma images, fabricate grades, train a model, or require a second copy of the datasets. Keep the earlier Gradio app as a fallback.

## Start on Windows (Git Bash)

Updating an already verified v2 installation with this appearance revision: stop the running server, extract the updated `retina_ui_v2` folder into `C:\CP2`, restart with `python retina_ui_v2/app.py`, then refresh the browser with Ctrl+F5. There are no dependency or checkpoint changes in this revision.

1. Extract the archive into the existing CP2 folder. The entry point should be `C:\CP2\retina_ui_v2\app.py`. The archive contains only the new `retina_ui_v2` directory.
2. Open Git Bash in the existing project and run:

   ```bash
   cd /c/CP2
   source .venv/Scripts/activate
   python retina_ui_v2/app.py --check
   ```

   This builds the interface and loads the accepted epoch-21 A3 checkpoint. It decodes no images and performs no inference. Expected final line: `RETINA UI V2 INPUT CHECK: PASS | A3 epoch 21 | Images decoded: 0`.

   If Gradio is missing, the included `python retina_ui_v2/setup_ui.py` uses the original version-constrained setup (Gradio 6.31.0). It refuses to replace an existing different Gradio version or upgrade an installed dependency. Resolve a version conflict before proceeding.
3. After the check passes, verify one fixed APTOS training image on the deployment GPU:

   ```bash
   python retina_ui_v2/app.py --verify
   ```

   The existing scientific check compares the exact preprocessing pixels with the canonical 384 × 384 image, then verifies the displayed grade and five probabilities against a direct forward pass. It checks frozen parameters and upstream file hashes. No cohort is re-evaluated. Each check saves a **new** timestamped report under `environment/data_audit/retina_ui_v2/`. These reports contain no uploaded pictures.
4. When verification passes, launch:

   ```bash
   python retina_ui_v2/app.py
   ```

   Open the localhost URL printed by the terminal (default `http://127.0.0.1:7860`). Use `--port 7862` if needed. Stop the server with Ctrl+C. The local project and CUDA-enabled environment must already be present. The interface intentionally stops on a failed checkpoint or software check; there is no demo-model fallback.

## First live acceptance

- Open **Use a sample → IDRiD sample → Analyse image**. If the local sample is missing, upload the original JPEG from the **segmentation testing** set, not a screenshot or a 384-pixel export. The saved localisation result for that exact study image predicts **grade 3**. Treat this as a cross-check, not its reference grade. The local reference grade is not needed to run the app.
- **Overview:** confirm the grade and all five probabilities appear. Percentages are rounded independently. NPDR means non-proliferative diabetic retinopathy. The APTOS sample (if available) is `000c1434d8d7`, a training illustration, never evidence of held-out performance.
- **Regional relevance:** switch **Full canvas → Retinal field only → Full canvas**. Only the displayed overlay should change. The original model input and all scores stay the same. An unavailable or flagged footprint disables the filtered view and gives the reason. The method uses the existing `experiments/aptos_fov_v1/fov_regions.py` function. It does not use the supplementary resegmentation experiment.
- **Region relationships:** live analysis displays the positive and negative GAT panels and signed values from this request. Links concern the predicted-class logit, not diagnosed lesions. An empty sign stays empty. If analysis fails, the result is cleared and a retry is offered.
- **Responsive layout:** the workspace uses light surfaces and dark text with either browser theme. The three equal-width result tabs stay below the fixed header while scrolling through a result. On narrow screens the labels shorten to Overview, Relevance and Relationships; the destinations keep their full names. On desktop the grade and probability cards have equal height and their headings align. Check both tables, the fixed footer and the About window at desktop and phone widths. Descriptions and filenames may wrap; control labels, percentages and region pairs should fit on one line.
- **Action feedback:** the ring appears at the centre of the viewport. Choosing a file opens the native picker; cancelling leaves the previous result in place. Selecting a file, loading a sample, analysing, and clearing wait for their own Gradio response. Tabs and the two equal-size relevance controls give brief feedback without a fixed delay or a new analysis. Model notes open in the centre of the viewport and scroll internally if needed. A reported transport error releases the overlay and offers a fresh workspace.
- **Error recovery:** after a valid result, try a corrupt PNG, a non-image, and a file larger than 20 MiB. The previous result must disappear. Clear the image and confirm the next analysis starts clean. Browser and operating-system file choosers may prefilter unsupported extensions.
- **Local handling:** file validation checks format, dimensions and decodability; it cannot confirm clinical gradability. App state is per browser session. Clear removes the displayed results and prepared input, not an immediate disk wipe. A private temporary cache is cleaned periodically (60-second interval; 900-second file age) and on a normal exit. Check Windows shutdown/cleanup locally.

## Implementation and scope

`app.py` wires the upload and run controls to Gradio. `_baseline/prototype_core_v1.py` is the original frozen adapter, copied without modification. `evidence.py` passes through the existing relationship record and masks a copy of the native-size overlay. `views.py`, `interface.js` and `design.css` implement the UI. Changing result tabs or the display mask is browser-side and never triggers a new forward pass. Neither checkpoint nor private uploads are included. All default network serving is loopback only.

See `VALIDATION.md` for what was checked before delivery and what still depends on your Windows installation.
