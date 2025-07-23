# Smart PPE Detection: AI-Based Segmentation for Construction Worker Safety

This repository provides an **AI-driven solution** for detecting and segmenting **Personal Protective Equipment (PPE)** and other essential safety objects on construction sites. The goal is to help ensure worker safety by automatically identifying critical PPE items, machinery, vehicles, and other objects crucial for safe construction site management.

This project leverages **state-of-the-art segmentation models** to identify and segment various objects in construction environments to enhance safety measures and compliance.
## Directory Structure


## File Descriptions

- **src/distributed_train.py**: Main script for distributed training of the Mask R-CNN model, leveraging cloud resources (AWS) for accelerated training.
- **src/inference.py**: Script for real-time PPE detection, used in safety systems for construction firms.
- **src/evaluate.py**: Evaluates model performance, computing metrics like mean Average Precision (mAP) to achieve 91% accuracy.
- **src/models/mask_rcnn.py**: Defines the custom Mask R-CNN model for PPE segmentation.
- **src/data/dataset.py**: Enhanced dataset class for loading and preprocessing PPE images with segmentation annotations.
- **src/data/transforms.py**: Custom data augmentations (e.g., resizing, flipping) to improve model robustness.
- **src/utils/distributed.py**: Helper functions for distributed training (e.g., multi-GPU or multi-node communication).
- **src/utils/logging.py**: Configures logging for training metrics and debugging.
- **src/utils/metrics.py**: Calculates evaluation metrics (e.g., mAP) for model performance.
- **configs/base.yaml**: Hyperparameters for the Mask R-CNN model (e.g., learning rate, batch size).
- **configs/aws_config.yaml**: Configuration for AWS cloud deployment (e.g., instance types, cluster settings).
- **scripts/launch_aws.sh**: Shell script to launch distributed training on an AWS cluster.
- **scripts/docker_build.sh**: Shell script to build the Docker container for the training environment.
- **scripts/run_inference.sh**: Shell script for deploying real-time inference.
- **figures/PYLON_Archi.png**: Diagram of the PYLON architecture for documentation (e.g., LaTeX paper).
- **figures/CAM_Appendix.png**: Grad-CAM visualization for documentation (e.g., LaTeX paper).
- **checkpoints/model_best.pth**: Trained model weights for the best-performing Mask R-CNN model.
- **data/ppe_dataset/**: Directory for PPE image dataset and annotations (not included in repo; provide your own).
- **tests/test_dataset.py**: Unit tests for the dataset class.
- **tests/test_model.py**: Unit tests for the Mask R-CNN model.
- **Dockerfile**: Defines the Docker container environment for training.
- **requirements.txt**: Lists Python dependencies (e.g., PyTorch, torchvision).
- **README.md**: This file, providing setup and usage instructions.

## Setup Instructions

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt





## Key Segmentation Classes:
The following are the objects targeted for segmentation in this repository:

1. **Safety-cone**
2. **Person-with-helmet**
3. **Safety-Vest**
4. **Safety-gloves**
5. **Person-without-safety-gloves**
6. **Person-without-helmet**
7. **Vehicle**
8. **Machinery**
9. **Safety-shoes**
10. **Person-without-mask**
11. **Vehicle**
12. **Safety-glasses**
13. **Safety-mask**
14. **Safety-suit**
15. **Helmet**
16. **Person-without-safety-shoes**

These objects are segmented to help with safety compliance and prevent accidents by ensuring proper PPE usage.

---

## Request for Dataset Access:

The dataset used for this project contains annotated images from construction sites, detailing various safety gear and objects. As this dataset is **private**, please contact us to request access. You can reach us by opening an issue or emailing us directly at [iikhaan@yahoo.com] for dataset access and further instructions.

---
