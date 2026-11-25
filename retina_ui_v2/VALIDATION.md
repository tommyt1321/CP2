# Engineering acceptance · Retina Insight Gradio v2

The delivered UI ran in Chromium against the real Gradio 6.31.0 service, using the real preparation, Attention Rollout and GAT code. The engineering backend used an **untrained CPU A3** with the public original IDRiD segmentation/testing/IDRiD_61. Its scores are not project results and are deliberately excluded. The production launcher only accepts the previously verified A3 epoch-21 checkpoint and project environment.

| Check | Observed result |
| --- | --- |
| Responsive type and containers | At 320, 390, 768, 820, 1068, 1121, 1280 and 1440 CSS pixels, five screen states passed the checks for wrapping and clipping. Tests measured the visible text ranges inside controls, the longest DR label, percentage and table cells, page width, desktop card sizes, shared gutters, and fixed header/footer. |
| Navigation | Overview, Regional relevance and Region relationships switched correctly. Keyboard tab-list navigation, the About dialog, automatic result focus on a narrow screen and return to Full canvas worked. |
| Real input and browser output wiring | A valid original went through upload and preparation; its model-input panel was pixel-identical to the accepted 384 × 384 reference. The engineering run returned five probabilities and two signed overlays with matching ranked tables. No engineering predictions or images are bundled. |
| Display-only retinal field | All **81,767** pixels inside the estimate matched the full overlay; all **65,689** exterior pixels were black. The original input remained unchanged. Across all three tabs and both field transitions, the analysis callback ran **once** (that one computation performs the underlying model forwards). |
| Input replacement and recovery | Unsupported extension, a file over 20 MiB and corrupt data cleared prior result panels and disabled analysis. A long valid filename remained readable. Clear removed the input and result. An intentionally failed analysis left no stale prediction and permitted retry. |
| Missing evidence | A missing, invalid or mismatched footprint yields no masked copy. The full map remains available with an explanation. Empty signed lists remain empty. |
| Frozen adapter | The packaged `_baseline/prototype_core_v1.py` matches the existing v1 source byte-for-byte (SHA-256 `44acfc098fe969b073ee74502b63b001b1a595e3bb81e58288ccc2bd56df1433`). |

### Result display revision

In Gradio 6.31.0, a dark theme reproduced the pale `#F4F4F5` text visible in the Windows screenshots despite the white result cards. At the same scroll offset the previous result tabs were behind the fixed header. The updated presentation was checked in Chromium with Gradio set to light, set to dark, and following a dark system preference, at viewport widths of 320, 390, 768, 820, 951, 1068, 1280 and 1440 CSS pixels.

- The brand, class labels and percentages on white measured 13.03:1 contrast (previously 1.10:1 with the dark theme); the grade description on its tinted card measured 11.85:1. The signed table values, captions, footer, primary upload button and dialog text were also checked on their rendered backgrounds.
- The three tabs had equal widths and remained clickable directly below the header on scroll. Mouse and keyboard tab changes exposed the new panel heading. Compact tab names, grade labels and percentages fitted without horizontal scrolling. In the desktop two-column layout the grade and probability cards matched in size, and the two relationship images and table tops aligned.
- These checks used the production Gradio components and a private layout fixture. They did not run inference or evaluate a new prediction. The fixture is not distributed.

### Action feedback, display controls and modal

The two display controls previously differed by two CSS pixels in vertical position in Chromium. Both now have the same top, width and height in either selected state. The native About dialog is centred relative to the viewport rather than the page; it remained centred after scrolling at 1177 × 881, 390 × 844, 320 × 568 and 820 × 420. Short viewports allow the dialog content to scroll.

The production Gradio callbacks and browser UI were exercised with an untrained engineering A3 and deliberately slowed preparation and inference. The centred ring appeared on activation, remained through intermediate preparation and analysis messages, and cleared when the new result, ready, error or cleared state was presented. Repeated invalid input received a new completion marker even when the error text was identical. Tests covered both sample buttons, native file chooser cancellation, valid and corrupt uploads, failed inference, duplicate attempts while busy, the sample selector, mouse and keyboard navigation, and opening and dismissing the dialog. Local display and navigation controls did not trigger additional inference. Reduced motion kept the activity indicator static. An injected upload failure and a queue submission failure both released the overlay and offered reload. The test harness and its outputs are not included.

This browser QA does not establish diagnostic accuracy, clinical usability, or Windows/CUDA parity. The appearance update still needs a visual check in the user's Windows browser.

### SLIC boundaries in the relationship panels

The optional outlines use the GAT label map from the same analysis. For IDRiD_61, the independently reconstructed canonical input and 256-region partition were used with the ten pairs and rounded scores transcribed from the submitted Windows result. This checked drawing geometry, not a new checkpoint prediction. Before the links were drawn, only pixels at transitions between labels changed in the display copy. The original RGB, labels, pair data and original signed images remained unchanged; solid link, node and shared-border colours were preserved at the same rendered pixels. A 255-region test grid was accepted with its actual count. An incompatible grid disabled the optional display and left the original output available.

In the Gradio 6.31.0 browser test, one analysis with the untrained engineering backend was followed by toggling both signed panels, navigating away and back, changing the separate retinal-field setting and toggling back with the keyboard. Both ranked tables, the overview and the original signed image sources matched their initial values. There was no second analysis or browser POST from these display actions. The control fit at 1440, 390 and 320 CSS pixels. Clearing the workspace removed the switch and its maps.
