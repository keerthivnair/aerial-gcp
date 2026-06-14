import os
import json
import cv2
import shutil
from sklearn.model_selection import GroupKFold

def convert_to_yolo():
    base_dir = "data/yolo_dataset"
    dirs = [
        "images/train", "images/val",
        "labels/train", "labels/val"
    ]
    
    for d in dirs:
        os.makedirs(os.path.join(base_dir, d), exist_ok=True)

    with open('data/train/train_dataset/gcp_marks.json', 'r') as f:
        annotations = json.load(f)

    image_paths = [p for p in annotations.keys() if os.path.exists(os.path.join('data/train/train_dataset', p))]
    groups = [os.path.dirname(p) for p in image_paths]
    
    gkf = GroupKFold(n_splits=5)
    train_idx, val_idx = next(gkf.split(image_paths, groups=groups))
    
    train_paths = set([image_paths[i] for i in train_idx])
    
    BOX_SIZE = 150.0 

    for rel_path in image_paths:
        full_path = os.path.join('data/train/train_dataset', rel_path)
        img = cv2.imread(full_path)
        if img is None:
            continue
            
        h, w, _ = img.shape
        
        true_x = float(annotations[rel_path]['mark']['x'])
        true_y = float(annotations[rel_path]['mark']['y'])
        
        norm_x = true_x / w
        norm_y = true_y / h
        norm_w = BOX_SIZE / w
        norm_h = BOX_SIZE / h
        
        yolo_label_str = f"0 {norm_x} {norm_y} {norm_w} {norm_h}\n"
        
        split = "train" if rel_path in train_paths else "val"
        
        safe_filename = rel_path.replace('/', '_').replace(' ', '_')
        
        dest_img_path = os.path.join(base_dir, "images", split, safe_filename)
        shutil.copy(full_path, dest_img_path)
        
        txt_filename = safe_filename.rsplit('.', 1)[0] + '.txt'
        dest_txt_path = os.path.join(base_dir, "labels", split, txt_filename)
        
        with open(dest_txt_path, 'w') as f:
            f.write(yolo_label_str)

    yaml_content = f"""path: {os.path.abspath(base_dir)}
train: images/train
val: images/val

names:
  0: gcp
"""
    with open(os.path.join(base_dir, 'dataset.yaml'), 'w') as f:
        f.write(yaml_content)

if __name__ == '__main__':
    convert_to_yolo()
