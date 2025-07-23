# Smart PPE Detection: AI-Based Segmentation for Construction Worker Safety

This repository provides an **AI-driven solution** for detecting and segmenting **Personal Protective Equipment (PPE)** and other essential safety objects on construction sites. The goal is to help ensure worker safety by automatically identifying critical PPE items, machinery, vehicles, and other objects crucial for safe construction site management.

This project leverages **state-of-the-art segmentation models** to identify and segment various objects in construction environments to enhance safety measures and compliance.
ppe-detection-distributed/
ppe-detection-distributed/
├── src/
│   ├── distributed_train.py      (this file)
│   ├── models/
│   │   ├── mask_rcnn.py          # Custom model definition
│   ├── data/
│   │   ├── dataset.py            # Enhanced dataset class
│   │   ├── transforms.py         # Custom augmentations
│   ├── utils/
│   │   ├── distributed.py        # Helper functions
│   │   ├── logging.py            # Logging config
├── configs/
│   ├── base.yaml                 # Hyperparameters
│   ├── aws_config.yaml           # Cloud deployment
├── scripts/
│   ├── launch_aws.sh             # AWS cluster script
│   ├── docker_build.sh           # Container build
├── Dockerfile                    # Training environment
├── requirements.txt              # Python dependencies
└── README.md                     # Detailed setup guide
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
