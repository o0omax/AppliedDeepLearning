
import json
import os
import shutil

def filter_coco_annotations(input_file, output_file, target_categories, images_dir, output_images_dir):
    with open(input_file, 'r') as f:
        data = json.load(f)

    categories = data.get('categories', [])
    category_name_to_id = {c['name']: c['id'] for c in categories}

    target_category_ids = set()
    for c in target_categories:
        if c in category_name_to_id:
            target_category_ids.add(category_name_to_id[c])
        else:
            print(f"Warning: Category '{c}' not found.")

    annotations = data.get('annotations', [])
    filtered_annotations = [ann for ann in annotations if ann['category_id'] in target_category_ids]

    valid_image_ids = {ann['image_id'] for ann in filtered_annotations}
    images = data.get('images', [])
    filtered_images = [img for img in images if img['id'] in valid_image_ids]
    filtered_categories = [cat for cat in categories if cat['id'] in target_category_ids]

    filtered_data = {'images': filtered_images, 'annotations': filtered_annotations, 'categories': filtered_categories}

    os.makedirs(output_images_dir, exist_ok=True)
    for img in filtered_images:
        src_path = os.path.join(images_dir, img['file_name'])
        dst_path = os.path.join(output_images_dir, img['file_name'])
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)

    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=4)

def run_filter_coco(params):
    filter_coco_annotations(
        input_file=params['input_file'],
        output_file=params['output_file'],
        target_categories=params['target_categories'],
        images_dir=params['images_dir'],
        output_images_dir=params['output_images_dir']
    )
