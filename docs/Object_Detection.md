
# Computer Vision
Classification -> OD -> Segementation -> Tracking


## Output from an Object Detection Framework
- Class of Object
- Confidence Score
- Bounding Box/Anchor Box ((x, y), l, w)
  - How to build a Rectangle
    - Center of Rectangle (x, y)
    - Length of Rectangle (l)
    - Width of Rectangle (w)
- Number of Instances   

## Object Detection Framework
- [Tf1_OD](https://github.com/tensorflow/models/tree/v1.13.0/research/object_detection)
- Tf2_OD
- Detectron
- Yolo


## [Model Zoo](https://github.com/tensorflow/models/blob/v1.13.0/research/object_detection/g3doc/detection_model_zoo.md)
- Detection Model
  - SSD Family (light weight models): Process More Frames in One Pass
    - SSD
    - Yolo
  - RCNN Family (Heavy models)
    - RCNN (deprecated)
    - Fast RCNN (deprecated)
    - Fatser RCNN
    - Mask RCNN (Used for Segmentation Tasks)
  - CenterNet Family
- Classification Network (VGG/Inception/ResNet)
- Datatset: [COCO dataset](https://cocodataset.org/#explore)
  - 90 Classes

## Metrics
- mAP: Metric for Object Detection
- Speed(ms): How many Frames/Images a model will process per second (fps)
- TradeOff
  - if mAp is higher, speed will be lower
  - if speed is higher, mAP will be lower
