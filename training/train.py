import os
import argparse
import subprocess

def train_yolo(weights, data, epochs, batch_size, img_size, project, name, hyp=None, optimizer=None, cache=False):
    """Runs YOLOv5 training with the given parameters."""
    cmd = [
        "python", "yolov5/train.py",
        "--weights", weights,
        "--data", data,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--img", str(img_size),
        "--project", project,
        "--name", name
    ]
    if hyp:
        cmd.extend(["--hyp", hyp])
    if optimizer:
        cmd.extend(["--optimizer", optimizer])
    if cache:
        cmd.append("--cache")
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def validate_yolo(weights, data, img_size):
    """Runs YOLOv5 validation."""
    cmd = [
        "python", "yolov5/val.py",
        "--weights", weights,
        "--data", data,
        "--img", str(img_size)
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv5 training pipeline for defect detection.")
    parser.add_argument("--approach_b", action="store_true", help="Enable Approach B (Intermediate Transfer).")
    args = parser.parse_args()

    # Relative paths
    yolo_weights = "yolov5/models/yolov5n.yaml"
    data_aircraft = "./hyp/defect_detection.yaml"
    data_car = "./hyp/car_ddv2.yaml"
    hyp_aircraft = "./hyp/hyp_aircraft.yaml"
    project_path = "./experiments"

    if args.approach_b:
        print("Running Approach B: COCO → CarDD → Airplane")

        # Step 1: COCO → CarDD
        train_yolo(
            weights=yolo_weights,
            data=data_car,
            epochs=50,
            batch_size=16,
            img_size=640,
            project=project_path,
            name="cardd_from_coco"
        )

        # Step 2: CarDD → Airplane
        weights_cardd = os.path.join(project_path, "cardd_from_coco", "weights", "best.pt")
        train_yolo(
            weights=weights_cardd,
            data=data_aircraft,
            epochs=50,
            batch_size=16,
            img_size=640,
            project=project_path,
            name="airplane_from_cardd"
        )

        validate_yolo(
            weights=os.path.join(project_path, "airplane_from_cardd", "weights", "best.pt"),
            data=data_aircraft,
            img_size=640
        )
    else:
        print("Running Approach A: Direct Transfer COCO → Airplane")

        # First training
        train_yolo(
            weights=yolo_weights,
            data=data_aircraft,
            epochs=50,
            batch_size=16,
            img_size=640,
            project=project_path,
            name="airplane_from_coco"
        )

        # Second training with extended epochs
        train_yolo(
            weights=yolo_weights,
            data=data_aircraft,
            epochs=100,
            batch_size=16,
            img_size=640,
            project=project_path,
            name="airplane_from_coco_100"
        )

        # Third training with adjusted hyperparameters
        train_yolo(
            weights=yolo_weights,
            data=data_aircraft,
            epochs=100,
            batch_size=16,
            img_size=640,
            project=project_path,
            name="airplane_from_coco_adjusted",
            hyp=hyp_aircraft,
            optimizer="Adam",
            cache=True
        )

        validate_yolo(
            weights=os.path.join(project_path, "airplane_from_coco_adjusted", "weights", "best.pt"),
            data=data_aircraft,
            img_size=640
        )

if __name__ == "__main__":
    main()
