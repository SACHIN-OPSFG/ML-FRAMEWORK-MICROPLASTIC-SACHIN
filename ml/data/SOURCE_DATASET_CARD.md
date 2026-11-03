---
license: cc-by-4.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
  - split: test
    path: data/test-*
dataset_info:
  features:
  - name: image
    dtype: image
  - name: objects
    struct:
    - name: bbox
      list:
        list: float32
        length: 4
    - name: category_id
      list: int64
    - name: segmentation
      list:
        list:
          list: float32
    - name: area
      list: float32
  splits:
  - name: train
    num_bytes: 910766614
    num_examples: 1406
  - name: validation
    num_bytes: 202524227
    num_examples: 302
  - name: test
    num_bytes: 202915753
    num_examples: 302
  download_size: 1314271478
  dataset_size: 1316206594
---

# Dataset Card for Microplastic-Fragments-WWTP-dataset

## Dataset Summary
The **Microplastic-Fragments-WWTP-dataset** is a specialized computer vision dataset designed to train and evaluate deep learning models for the detection and segmentation of microplastic (MP) fragments in highly complex real-world environments. 

* The dataset contains 2010 annotated high-resolution images capturing diverse sludge treatment stages.
* Images were acquired from the sludge line of an operating Wastewater Treatment Plant (WWTP) located in Valencia, Spain.
* Annotations include both bounding boxes and polygonal segmentation masks to support single-stage detectors, two-stage frameworks, and foundation models.
* This dataset addresses a critical gap in environmental monitoring by providing images of fragments against complex organic backgrounds rather than ideal laboratory conditions.

## Data Collection & Sample Preparation
The presence of organic matter in sewage sludge complicates the direct visualization of microplastics. To isolate the fragments, samples underwent a strict chemical conditioning protocol:

* **Oxidation:** Samples were mixed with H2O2 (30%) by mixing 25 mL of the reactive with 25 mL of the sample, then stirred for 2 hours at 60°C to reduce organic matter.
* **Filtration:** The oxidized samples were filtered using cellulose acetate filters with a pore size of 5 µm and a diameter of 90 mm. To prevent fragments from being masked by residual unoxidized material, sample contents were distributed across multiple filters based on their Total Solids (TS) concentration.
* **Drying:** Filters were dried at 60°C for 2.5 hours.
* **Image Acquisition:** Filters were photographed using a Leica S APO Zoom 80:10x stereomicroscope connected to LAS X1 software. Images were captured at 42x, 32x, 20x, and 16x magnifications, and a scale legend was embedded in the top-right corner of each image.

## Dataset Structure
The dataset is provided as a Hugging Face `DatasetDict` and is downloaded as four Parquet files. It is divided into pre-defined `train`, `validation`, and `test` splits. 

Within each split, the dataset contains the following features:
* `image`: The decoded visual representation of the microplastic fragments (originally 3840x2160 pixel `.jpg` images).
* `objects`: A sequence of ground-truth annotations associated with the corresponding image. This is structured as a dictionary containing:
  * **Bounding Boxes:** Formatted in the COCO standard `[xmin, ymin, width, height]`.
  * **Segmentation Masks:** Polygonal points formatted as `[x0, y0, x1, y1, ..., xn, yn]`.
  * **Category Labels & Areas:** Category identifications and area calculations for every fragment.
  * * **Class Mapping & Indexing:** 
    * `category_id: 1` → Foreground target class (`Microplastics` / `fragment`).
    * `category_id: 0` → Implicit background. Background tiles contain no annotated instances (`labels: []`), which prevents standard torchvision, Detectron2, and Mask R-CNN index collisions where `0` is strictly reserved for the background class.

## Exploratory Data Analysis
* **Fragment Frequency:** 50% of the images contain exactly one fragment, and 90% feature four or fewer fragments.
* **Morphological Diversity:** The dataset captures a wide scale of fragments, with the relative bounding box area representing up to 30% of the total image surface for the majority of instances. 

## Usage Notes & Limitations
* **Bias:** Annotations were made manually by specialized personnel; consequently, distinguishing highly degraded plastics from surrounding debris involves some subjective human judgment.
* **Scope:** The dataset is specifically curated to feature microplastic fragments. While other microplastic morphologies are completely excluded from bounding box and polygon annotations, they may occasionally be present as unannotated background elements.
* **Generalization:** Models trained solely on this dataset may require careful evaluation before deployment on samples obtained from different geographical locations (e.g., marine environments) or processed via different chemical protocols.