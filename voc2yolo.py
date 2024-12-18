import os
import json
import shutil

# Paths
base_path = "/mnt/c/Users/maxim/Downloads/CarDD_release/CarDD_release/CarDD_COCO"
sets = ['filtered_train2017', 'filtered_val2017', 'filtered_test2017']  # Dataset subsets
output_sets = ['filtered_and_yolo_train2017', 'filtered_and_yolo_val2017', 'filtered_and_yolo_test2017']
valid_category_ids = {1: 0, 3: 1}  # Map COCO category IDs (1, 3) to YOLO class IDs (0, 1)
classes = ['dent', 'crack']  # Class names


def convert_bbox(size, bbox):
    """Convert COCO bbox format [x, y, width, height] to YOLO format [x_center, y_center, width, height]."""
    x, y, w, h = bbox
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (x + w / 2.0) * dw
    y_center = (y + h / 2.0) * dh
    w = w * dw
    h = h * dh
    return x_center, y_center, w, h


# Process each dataset
for set_name, output_set_name in zip(sets, output_sets):
    print(f"\n🔍 Processing {set_name}...")

    # Paths
    images_dir = os.path.join(base_path, set_name)
    annotations_file = os.path.join(base_path, "annotations",
                                    f"filtered_instances_{set_name.replace('filtered_', '')}.json")
    output_images_dir = os.path.join(base_path, output_set_name, "images")
    output_labels_dir = os.path.join(base_path, output_set_name, "labels")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # Load COCO annotations
    with open(annotations_file, 'r') as f:
        data = json.load(f)

    # Create image ID to filename map
    image_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
    image_id_to_size = {img['id']: (img['width'], img['height']) for img in data['images']}

    # Group annotations by image ID
    anns_by_image = {}
    for ann in data['annotations']:
        if ann['category_id'] in valid_category_ids:  # Filter out invalid categories
            image_id = ann['image_id']
            anns_by_image.setdefault(image_id, []).append(ann)

    # Convert annotations and copy images
    for image_id, annotations in anns_by_image.items():
        # File paths
        filename = image_id_to_filename[image_id]
        src_image_path = os.path.join(images_dir, filename)
        dst_image_path = os.path.join(output_images_dir, filename)
        label_file_path = os.path.join(output_labels_dir, filename.replace('.jpg', '.txt'))

        # Copy the image
        if not os.path.exists(src_image_path):
            print(f"❗ Warning: Missing image file '{src_image_path}'")
            continue
        shutil.copy2(src_image_path, dst_image_path)

        # Write annotations in YOLO format
        with open(label_file_path, 'w') as label_file:
            for ann in annotations:
                class_id = valid_category_ids[ann['category_id']]  # Map category_id to YOLO class_id
                bbox = ann['bbox']
                img_size = image_id_to_size[image_id]
                yolo_bbox = convert_bbox(img_size, bbox)
                label_file.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")

        print(f"  ✅ Processed image: {filename} -> {label_file_path}")

print("\n✅ COCO to YOLOv5 conversion completed successfully!")
