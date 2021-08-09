
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
  - if mAP is higher, speed will be lower
  - if speed is higher, mAP will be lower

## [Setup Tf1.4 environment](https://pastebin.com/YDgbqzTx)
1. Create Conda environment and install libraries
    ```
    # Creating virtual env using conda
    conda create -n tf14env python=3.6.9
    conda activate tf14env

    # pypi 
    pip install pillow lxml Cython contextlib2 jupyter matplotlib pandas opencv-python tensorflow==1.14.0
    ```
2. Framework and Model setup
  - [Downlaod Framework Repository](https://github.com/tensorflow/models/tree/v1.13.0)
    - provide models/models/research directory
  - [Select and Download a Pre Trained Model from Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md)
    - example [faster_rcnn_inception_v2_coco_2018_01_28](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)
  - [Download Utils](https://drive.google.com/file/d/12F5oGAuQg7qBM_267TCMt_rlorV-M7gf/view)
3. Install **setup.py** from **Research** folder
    ```
    python setup.py install #install object detection
    ```
4. Protobuff to **py** conversion
    ```
    conda install -c anaconda protobuf
    # linux mac
    protoc object_detection/protos/*.proto --python_out=.
    #windows
    protoc object_detection/protos/*.proto --python_out=.
    ```
5. Open the **object_detection_tutorial.ipynb**
    - Parent Directory (models/model/research)
    - !conda env list ("*" shows current active env)
    - You might have to move object_detection_tutorial.ipynb to research directory
      ```
      %matplotlib inline
      plt.figure(figsize=(200,200))
      plt.imshow(image_np)
      ```
6. Copy the file **xml_to_csv** from utils folder to models repo in reaserch folder
   ```
   python xml_to_csv.py (tensorflow1/models/research/object_detection)
   ```
7. Copy the file **tfgenerate.record** from utils folder to models repo in reaserch folder
   ```
   python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
   python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
   ```
8. Copy the file **train.py** from legacy folder in object_detection to research, Run below command from research folder
   ```
   python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_coco.config
   ```
9. Exporting Inference Graph
    - Replace the XXXX with the last generated ckpt file inside the training folder
    ```
    python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix training/model.ckpt-1000 --output_directory inference_graph
    ```

## How to Open Camera
  ```
  import cv2

  cap = cv2.VideoCapture(0)
  with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
      while True:
        ret, image_np = cap.read()
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections],feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

        cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break

  cap.release()
  ```
