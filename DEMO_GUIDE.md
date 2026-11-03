# Five-minute demonstration

1. Start with `START_PROJECT.bat` and open http://127.0.0.1:8000.
2. Explain: "This system screens microscopy images for suspected microplastic fragments using a trained YOLOv8 object detector."
3. Click **Try a real test sample**. Keep confidence at 25%. Leave calibration empty unless your microscope calibration is known.
4. Click **Analyze sample**. Show the annotated image and explain the fragment bounding boxes.
5. Show total count and the corresponding table rows. Explain confidence as a model score and dimensions as pixels.
6. Download the PDF report. Show the image, particle measurements and interpretation notes.
7. Open **Analysis history** and reopen the saved sample.
8. Open **Model & evaluation**. Show the real held-out test precision, recall and mAP values. Do not call confidence or mAP overall "accuracy."
9. Open `ml/notebooks/microplastic_yolov8_training.ipynb` in VS Code/Colab to show dataset inspection, training source, evaluation curves and inference.

## Answers to likely questions

**Which algorithm?** YOLOv8n transfer learning for object detection, with OpenCV image handling and Python particle analysis. This delivered version uses trained neural-network weights, not a contour demonstration.

**How many classes?** One annotated class: `microplastic_fragment`. Class names come from the dataset configuration and model metadata. Other morphologies were not labeled in this dataset.

**Dataset size?** 2,010 images: 1,406 train, 302 validation, 302 test. Source and transformations are documented in `DATASET_ATTRIBUTION.md`.

**How was data leakage checked?** Publisher splits were preserved and exact source-image hashes were compared. No exact duplicates were found. Near-duplicate acquisition images still require a stronger domain/group analysis.

**Does it prove the object is plastic?** No. It predicts an image annotation class. Chemical confirmation requires appropriate laboratory methods outside this project.

**What does Low mean?** Fewer than 10 detections in this image at the chosen threshold. It is a configurable demonstration band, not a safe-water designation or concentration.

**Can it measure micrometers?** Only with a valid microscope calibration supplied by the user. The model does not infer calibration from an arbitrary image.

**Can Sachin run it without a GPU?** Yes. Application inference runs on CPU. A GPU is useful for retraining, including through Google Colab.

**Can he run it without internet?** After the first Python dependency installation, the shipped dashboard, model, samples, fonts and reports run locally. Dataset downloads and retraining from missing pretrained weights may require internet.
