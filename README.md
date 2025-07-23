# PPE Detection with Distributed Training

## Directory Structure

```
Folder PATH listing for volume Data
Volume serial number is 0898-A058
G:.
│   README.md
│   requirements.txt
│   train_ddp.py
│   
├───src
│       start_train.py
│       
└───ultralytics
    ├───cfg
    │   │   default.yaml
    │   │   __init__.py
    │   │   
    │   ├───models
    │   │   └───v8
    │   │           yolov8_seg.yaml
    │   │           yolov8_seg_ECA.yaml
    │   │           yolov8_seg_GAM.yaml
    │   │           yolov8_seg_GCT_M1.yaml
    │   │           yolov8_seg_GCT_M2.yaml
    │   │           yolov8_seg_GCT_M3.yaml
    │   │           yolov8_seg_GC_M1.yaml
    │   │           yolov8_seg_GC_M2.yaml
    │   │           yolov8_seg_GC_M3.yaml
    │   │           yolov8_seg_GE_M1.yaml
    │   │           yolov8_seg_GE_M2.yaml
    │   │           yolov8_seg_GE_M3.yaml
    │   │           yolov8_seg_ResBlock_CBAM.yaml
    │   │           yolov8_seg_SA.yaml
    │   │           yolov8_SE_M1.yaml
    │   │           yolov8_SE_M2.yaml
    │   │           yolov8_SE_M3.yaml
    │   │           
    │   └───trackers
    │           botsort.yaml
    │           bytetrack.yaml
    │           
    ├───data
    │   │   annotator.py
    │   │   augment.py
    │   │   base.py
    │   │   build.py
    │   │   converter.py
    │   │   dataset.py
    │   │   loaders.py
    │   │   utils.py
    │   │   __init__.py
    │   │   
    │   ├───dataloaders
    │   │       __init__.py
    │   │       
    │   └───scripts
    │           download_weights.sh
    │           get_coco.sh
    │           get_coco128.sh
    │           get_imagenet.sh
    │           
    ├───engine
    │       exporter.py
    │       model.py
    │       predictor.py
    │       results.py
    │       trainer.py
    │       validator.py
    │       __init__.py
    │       
    ├───hub
    │       auth.py
    │       session.py
    │       utils.py
    │       __init__.py
    │       
    └───models
        │   __init__.py
        │   
        ├───fastsam
        │       model.py
        │       predict.py
        │       prompt.py
        │       utils.py
        │       val.py
        │       __init__.py
        │       
        ├───nas
        │       model.py
        │       predict.py
        │       val.py
        │       __init__.py
        │       
        ├───rtdetr
        │       model.py
        │       predict.py
        │       train.py
        │       val.py
        │       __init__.py
        │       
        ├───sam
        │   │   amg.py
        │   │   build.py
        │   │   model.py
        │   │   predict.py
        │   │   __init__.py
        │   │   
        │   └───modules
        │           decoders.py
        │           encoders.py
        │           sam.py
        │           tiny_encoder.py
        │           transformer.py
        │           __init__.py
        │           
        ├───utils
        │       loss.py
        │       ops.py
        │       __init__.py
        │       
        └───yolo
            │   model.py
            │   __init__.py
            │   
            ├───classify
            │       predict.py
            │       train.py
            │       val.py
            │       __init__.py
            │       
            ├───detect
            │       predict.py
            │       train.py
            │       val.py
            │       __init__.py
            │       
            ├───pose
            │       predict.py
            │       train.py
            │       val.py
            │       __init__.py
            │       
            └───segment
                    predict.py
                    train.py
                    val.py
                    __init__.py
                    
```
