# Actual model evaluation

Completed stable training: **40 epochs** (maximum 40), YOLOv8n, image size 640, batch 12, seed 42, mixed precision disabled, 4 data-loader workers.

The delivered weights are selected using validation performance. Test data is not used to train or select checkpoints. All values below were computed by running the delivered model on the supplied data partitions.

| Split | Images | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| val | 302 | 0.7645 (76.45%) | 0.7772 (77.72%) | 0.8527 (85.27%) | 0.6770 (67.70%) |
| test | 302 | 0.7997 (79.97%) | 0.7636 (76.36%) | 0.8535 (85.35%) | 0.6893 (68.93%) |

## Evidence

- `ml/evaluation/metrics.json`: raw measured values and per-image evaluation timings.
- `ml/evaluation/test`: held-out precision-recall curves, confusion matrices and prediction images.
- `ml/evaluation/val`: validation figures.
- `ml/runs/fragment_detector/results.csv` and `results.png`: stable training progress.
- `ml/models/best.pt` and `last.pt`: finite trained model parameters verified before delivery.

The original source provided 1,406 train / 302 validation / 302 test images. The only target class is `microplastic_fragment`. These scores quantify agreement with dataset annotations, not chemical confirmation or performance on all water samples. Confidence values in the app are per-detection scores; they are not an overall accuracy metric.

An earlier mixed-precision resumed training attempt produced non-finite values and was discarded. The delivered model was trained afresh with mixed precision disabled. No metrics or weights from that failed attempt are included in these results.

Evaluation speed is GPU-based for this training run. The dashboard uses CPU inference and reports its measured per-request time separately.
