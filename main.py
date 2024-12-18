import argparse
import json
from filter_coco import run_filter_coco
from voc2yolo import convert_coco_to_yolo

def main():
    parser = argparse.ArgumentParser(description="Run COCO filtering and YOLO conversion pipeline.")
    parser.add_argument("--operation", choices=['filter', 'convert'], required=True, help="Choose the operation to run.")
    parser.add_argument("--config", required=True, help="Path to the configuration JSON file.")
    parser.add_argument("--preprocess", action='store_true', help="Run all preprocessing steps if set to True.")
    args = parser.parse_args()

    # Load parameters from config
    with open(args.config, 'r') as f:
        params = json.load(f)

    if args.preprocess:
        print("Preprocessing enabled. Running all operations...")
        print("Running COCO Filtering...")
        run_filter_coco(params)
        print("COCO Filtering completed.")

        print("Running COCO to YOLO Conversion...")
        convert_coco_to_yolo(
            base_path=params['base_path'],
            sets=params['sets'],
            output_sets=params['output_sets'],
            valid_category_ids=params['valid_category_ids']
        )
        print("COCO to YOLO Conversion completed.")

    else:
        print("Preprocessing disabled. Running specific operation only...")
        if args.operation == 'filter':
            print("Running COCO Filtering...")
            run_filter_coco(params)
            print("COCO Filtering completed.")
        elif args.operation == 'convert':
            print("Running COCO to YOLO Conversion...")
            convert_coco_to_yolo(
                base_path=params['base_path'],
                sets=params['sets'],
                output_sets=params['output_sets'],
                valid_category_ids=params['valid_category_ids']
            )
            print("COCO to YOLO Conversion completed.")

if __name__ == "__main__":
    main()
