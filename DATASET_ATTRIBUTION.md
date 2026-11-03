# Dataset attribution and transformation record

**Microplastic-Fragments-WWTP-Dataset** by **andreusv** (the credited uploader/creator handle on the source repository).

- Source: https://huggingface.co/datasets/andreusv/Microplastic-Fragments-WWTP-Dataset
- DOI shown by the dataset page: https://doi.org/10.57967/hf/10280
- License: Creative Commons Attribution 4.0 International (CC BY 4.0).
- License text: https://creativecommons.org/licenses/by/4.0/
- Accessed: 3 October 2026.
- Dataset repository revision: `bdbb9dd898b7d96187e1d7e4f1a49babc85f3160` (pinned in the download script).

The source dataset card is included verbatim at `ml/data/SOURCE_DATASET_CARD.md`. The repository describes stereomicroscope images from an operating wastewater-treatment plant in Valencia, Spain, with manually annotated microplastic fragments in prepared sludge samples.

Changes for this project: original Parquet image bytes were decoded, converted to RGB JPEG, proportionally resized to a longest side of at most 1280 pixels and re-encoded at quality 92. COCO-format boxes were converted to normalized YOLO format. Foreground category 1 was mapped to YOLO class 0 (`microplastic_fragment`). Original train/validation/test membership was preserved. Original file hashes were audited for exact duplicates. Model training was carried out independently in this project; these metrics are not copied from a paper or repository.

When redistributing the ZIP or using figures in a report, keep this attribution and note these transformations. The publisher does not endorse this implementation. Dataset annotations describe fragments only; other microplastic shapes are outside its annotated scope.
