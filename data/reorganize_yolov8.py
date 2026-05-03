import shutil
from pathlib import Path

import yaml
from PIL import Image


def main():
    root = Path(__file__).parent.parent
    raw_dir = root / "data" / "raw"
    yolo_dir = raw_dir / "Chart-pattern.v2i.yolov8"
    processed_dir = root / "data" / "processed"

    if not yolo_dir.exists():
        print(f"YOLO directory {yolo_dir} does not exist.")
        return

    with open(yolo_dir / "data.yaml", "r") as f:
        data_info = yaml.safe_load(f)

    class_names = data_info.get("names", [])

    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "valid", "test"]:
        split_img_dir = yolo_dir / split / "images"
        split_lbl_dir = yolo_dir / split / "labels"

        if not split_img_dir.exists() or not split_lbl_dir.exists():
            continue

        split_processed_dir = processed_dir / split
        split_processed_dir.mkdir(parents=True, exist_ok=True)

        for img_path in split_img_dir.glob("*.*"):
            lbl_path = split_lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            with open(lbl_path, "r") as f:
                lines = f.readlines()

            if not lines:
                continue

            # For simplicity, if there are multiple objects, we just use the first one
            first_line = lines[0].strip().split()
            class_id = int(first_line[0])
            class_name = class_names[class_id]

            # Parse polygon coordinates
            coords = [float(x) for x in first_line[1:]]

            class_dir = split_processed_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # Open image, crop to bounding box, save
            try:
                img = Image.open(img_path).convert("RGB")
                w, h = img.size

                # Convert normalized coords back to pixel coords
                x_coords = [int(c * w) for c in coords[0::2]]
                y_coords = [int(c * h) for c in coords[1::2]]

                min_x = max(0, min(x_coords))
                max_x = min(w, max(x_coords))
                min_y = max(0, min(y_coords))
                max_y = min(h, max(y_coords))

                if max_x > min_x and max_y > min_y:
                    cropped = img.crop((min_x, min_y, max_x, max_y))
                    cropped.save(class_dir / img_path.name)
                else:
                    img.save(class_dir / img_path.name)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")


if __name__ == "__main__":
    main()
