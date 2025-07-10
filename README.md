# Concrete Crack Detection & Analysis

A complete pipeline for detecting, analyzing, and measuring concrete cracks using YOLOv8 and computer vision techniques.


## Features
- Dataset organization for YOLOv8
- YOLOv8 model training
- Crack detection and merging
- Semantic segmentation of cracks
- Precise crack measurements (length, width)
- Visualization of results

## Installation
1. Clone this repository
2. Install requirements:

## Dataset

The original image data is not included in this repository. To download the dataset:

- **Visit**: [USU Concrete Crack Dataset](https://digitalcommons.usu.edu/all_datasets/48/)
- **Download** the dataset files.
- **Extract** them into a `dataset` folder in the project root with this structure:


# Usage

### 1. Organize Dataset

```bash
python organize.py
```
This will create the YOLOv8 directory structure and split the dataset.

### 2. Train Model
```bash
python train.py
```
Trains the YOLOv8 detection model using the prepared dataset.

### 3. Analyze Images
```bash
python analyze.py --image test_images/1.jpg
```
Performs crack detection and analysis on the specified image.
