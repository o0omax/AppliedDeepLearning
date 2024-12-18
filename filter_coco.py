import json
import os
import shutil


def filter_coco_annotations(input_file, output_file, target_categories, images_dir, output_images_dir):
    # Load the JSON data
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Extract categories
    categories = data.get('categories', [])
    # Map category name to ID
    category_name_to_id = {c['name']: c['id'] for c in categories}

    # Find the category IDs for the target categories
    target_category_ids = set()
    for c in target_categories:
        if c in category_name_to_id:
            target_category_ids.add(category_name_to_id[c])
        else:
            print(f"Warning: Category '{c}' not found in {input_file} categories. It will be ignored.")

    # Filter annotations by target categories
    annotations = data.get('annotations', [])
    filtered_annotations = [ann for ann in annotations if ann['category_id'] in target_category_ids]

    # Get the set of image IDs corresponding to valid annotations
    valid_image_ids = set(ann['image_id'] for ann in filtered_annotations)

    # Filter images to only those which contain target categories
    images = data.get('images', [])
    filtered_images = [img for img in images if img['id'] in valid_image_ids]

    # Filter categories to only the target categories
    filtered_categories = [cat for cat in categories if cat['id'] in target_category_ids]

    # Create filtered data
    filtered_data = {
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': filtered_categories
    }

    # Ensure output directory for images exists
    os.makedirs(output_images_dir, exist_ok=True)

    # Copy only the filtered images to the new directory
    for img in filtered_images:
        file_name = img['file_name']
        src_path = os.path.join(images_dir, file_name)
        dst_path = os.path.join(output_images_dir, file_name)

        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
        else:
            print(f"Warning: Image file '{file_name}' not found at '{src_path}'. Skipping.")

    # Save the filtered data
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=4)


if __name__ == "__main__":
    # Define input paths
    annotation_dir = "/mnt/c/Users/maxim/Downloads/CarDD_release/CarDD_release/CarDD_COCO/annotations"
    train_input = os.path.join(annotation_dir, "instances_train2017.json")
    val_input = os.path.join(annotation_dir, "instances_val2017.json")
    test_input = os.path.join(annotation_dir, "instances_test2017.json")

    # Define output paths
    train_output = os.path.join(annotation_dir, "filtered_instances_train2017.json")
    val_output = os.path.join(annotation_dir, "filtered_instances_val2017.json")
    test_output = os.path.join(annotation_dir, "filtered_instances_test2017.json")

    # Original images directories
    # Make sure these paths are correct and contain the images referenced by the annotations.
    train_images_dir = "/mnt/c/Users/maxim/Downloads/CarDD_release/CarDD_release/CarDD_COCO/train2017"
    val_images_dir = "/mnt/c/Users/maxim/Downloads/CarDD_release/CarDD_release/CarDD_COCO/val2017"
    test_images_dir = "/mnt/c/Users/maxim/Downloads/CarDD_release/CarDD_release/CarDD_COCO/test2017"

    # Filtered images directories (new folders)
    # These will be created if they don't exist
    filtered_train_images_dir = "/mnt/c/Users/maxim/Downloads/CarDD_release/CarDD_release/CarDD_COCO/filtered_train2017"
    filtered_val_images_dir = "/mnt/c/Users/maxim/Downloads/CarDD_release/CarDD_release/CarDD_COCO/filtered_val2017"
    filtered_test_images_dir = "/mnt/c/Users/maxim/Downloads/CarDD_release/CarDD_release/CarDD_COCO/filtered_test2017"

    # Target categories
    target_cats = ["dent", "crack"]

    # Filter the datasets and copy filtered images
    filter_coco_annotations(train_input, train_output, target_cats, train_images_dir, filtered_train_images_dir)
    filter_coco_annotations(val_input, val_output, target_cats, val_images_dir, filtered_val_images_dir)
    filter_coco_annotations(test_input, test_output, target_cats, test_images_dir, filtered_test_images_dir)

    print("Filtering complete. New JSON files and filtered image folders have been created:")
    print(train_output, "->", filtered_train_images_dir)
    print(val_output, "->", filtered_val_images_dir)
    print(test_output, "->", filtered_test_images_dir)
