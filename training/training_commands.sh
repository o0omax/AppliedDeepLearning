#!/bin/bash

# Relative paths
YOLO_WEIGHTS="yolov5/models/yolov5n.yaml"
DATA_AIRCRAFT="./hyp/defect_detection.yaml"
DATA_CAR="./hyp/car_ddv2.yaml"
HYP_AIRCRAFT="./hyp/hyp_aircraft.yaml"
PROJECT_PATH="./experiments"

if [ "$1" == "A" ]; then
    echo "Running Approach A: Direct Transfer COCO → Airplane"

    # First training
    python yolov5/train.py --weights $YOLO_WEIGHTS --data $DATA_AIRCRAFT --epochs 50 --batch-size 16 --img 640 --project $PROJECT_PATH --name airplane_from_coco

    # Second training with extended epochs
    python yolov5/train.py --weights $YOLO_WEIGHTS --data $DATA_AIRCRAFT --epochs 100 --batch-size 16 --img 640 --project $PROJECT_PATH --name airplane_from_coco_100

    # Third training with adjusted hyperparameters
    python yolov5/train.py --weights $YOLO_WEIGHTS --data $DATA_AIRCRAFT --epochs 100 --batch-size 16 --img 640 --project $PROJECT_PATH --name airplane_from_coco_adjusted --hyp $HYP_AIRCRAFT --optimizer Adam --cache

    # Validation
    python yolov5/val.py --weights $PROJECT_PATH/airplane_from_coco_adjusted/weights/best.pt --data $DATA_AIRCRAFT --img 640

elif [ "$1" == "B" ]; then
    echo "Running Approach B: COCO → CarDD → Airplane"

    # Step 1: COCO → CarDD
    python yolov5/train.py --weights $YOLO_WEIGHTS --data $DATA_CAR --epochs 50 --batch-size 16 --img 640 --project $PROJECT_PATH --name cardd_from_coco

    # Step 2: CarDD → Airplane
    python yolov5/train.py --weights $PROJECT_PATH/cardd_from_coco/weights/best.pt --data $DATA_AIRCRAFT --epochs 50 --batch-size 16 --img 640 --project $PROJECT_PATH --name airplane_from_cardd

    # Validation
    python yolov5/val.py --weights $PROJECT_PATH/airplane_from_cardd/weights/best.pt --data $DATA_AIRCRAFT --img 640

else
    echo "Usage: $0 [A|B]"
    echo "  A: Direct Transfer COCO → Airplane"
    echo "  B: Intermediate Transfer COCO → CarDD → Airplane"
fi
