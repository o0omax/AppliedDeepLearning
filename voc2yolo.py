
import os
import json
import shutil

def convert_coco_to_yolo(base_path, sets, output_sets, valid_category_ids):
    def convert_bbox(size, bbox):
        x, y, w, h = bbox
        dw = 1.0 / size[0]
        dh = 1.0 / size[1]
        return (x + w / 2) * dw, (y + h / 2) * dh, w * dw, h * dh

    for set_name, output_set_name in zip(sets, output_sets):
        images_dir = os.path.join(base_path, set_name)
        annotations_file = os.path.join(base_path, "annotations", f"{set_name}.json")
        output_images_dir = os.path.join(base_path, output_set_name, "images")
        output_labels_dir = os.path.join(base_path, output_set_name, "labels")

        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        with open(annotations_file, 'r') as f:
            data = json.load(f)

        for img in data['images']:
            src = os.path.join(images_dir, img['file_name'])
            dst = os.path.join(output_images_dir, img['file_name'])
            shutil.copy2(src, dst)

        for ann in data['annotations']:
            bbox = convert_bbox((img['width'], img['height']), ann['bbox'])
            with open(os.path.join(output_labels_dir, f"{img['file_name'].replace('.jpg', '.txt')}"), 'w') as f:
                f.write(f"{valid_category_ids[ann['category_id']]} {' '.join(map(str, bbox))}")
