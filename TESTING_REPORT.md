# Testing report

**24 real-model API integration checks passed.**

Environment: Python 3.10.18, Conda `new_env`, NVIDIA RTX 4060 Laptop GPU for training; CPU inference in the API. Dependency versions are recorded in `requirements-tested.txt`.

| Check | Result |
|---|---|
| Backend healthy with trained model | PASS |
| Model exposes actual evaluation metrics | PASS |
| Real held-out image accepted | PASS |
| Particle counts agree with records and classes | PASS |
| Every detection has valid bounds and confidence | PASS |
| Calibration measurements agree | PASS |
| Annotated image generated | PASS |
| PDF generated from saved analysis | PASS |
| Analysis persisted in history | PASS |
| No physical dimensions without calibration | PASS |
| Unsupported file rejected | PASS |
| Corrupt image rejected | PASS |
| Disguised unsupported image rejected | PASS |
| Negative calibration rejected | PASS |
| Nonfinite calibration rejected | PASS |
| Invalid confidence rejected | PASS |
| Oversized upload rejected | PASS |
| Oversized decoded image rejected | PASS |
| Zero detections handled safely | PASS |
| Unknown report returns 404 | PASS |
| Report ID cannot escape storage | PASS |
| Internal database is not downloadable | PASS |
| Configured frontend origin allowed | PASS |
| Missing model gives readable configuration error | PASS |

## Real sample evidence

- Supplied held-out sample: **4 detections** at confidence 0.25.
- Annotated image: `output/verification/sample_annotated.jpg`.
- Analysis JSON: `output/verification/sample_analysis.json`.
- Actual PDF report: `output/verification/sample_analysis_report.pdf`.
- The shareable sample report uses pixel dimensions; the 0.5 micrometers/pixel value was a numeric calibration test fixture only.

## Notebook execution

- `microplastic_python_implementation.ipynb`: 9 code cells executed successfully in `new_env`.
- `microplastic_yolov8_training.ipynb`: 8 code cells executed successfully in `new_env`.

## Frontend and portability

- React/Vite production build completed successfully.
- Browser checks: sample selection, analysis, result cards/table/chart, PDF download, history reopening and model metrics view.
- Desktop and mobile screenshots are in `output/playwright`.
- Local fonts and the compiled frontend are bundled for offline operation after dependency setup.
- PDF pages were rendered and visually inspected for clipping and overlap.
- ZIP file CRC and manifest integrity verified; the extracted package was tested with the same Python environment.

## Verification boundaries

Testing was performed locally and on the extracted ZIP. A separate physical laptop and a Google Colab GPU session were not available for verification. The notebooks contain portable setup and training code; exact training results can vary on other hardware. CORS is configured for local Vite development. Missing-model, zero-result and malformed-input conditions were tested explicitly.

The application is intended for a local academic demonstration. It has no public multi-user authentication layer.
