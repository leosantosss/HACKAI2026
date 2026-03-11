import json
import os
import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import sys

# Add src to path if needed for config/id2label logic (though not used here directly yet)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

def verify():
    coco_path = "data/vizzion-1/train/_annotations.coco.json"
    img_dir = "data/vizzion-1/train/"
    
    with open(coco_path) as f:
        data = json.load(f)
        
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Model mapping (id2label from id2label.json)
    with open("models/id2label.json") as f:
        id2label = json.load(f)
    label2id = {v: int(k) for k, v in id2label.items()}
    
    # Create the mapping: COCO_ID -> MODEL_ID
    coco_to_model = {}
    for coco_id, name in categories.items():
        if name in label2id:
            coco_to_model[coco_id] = label2id[name]
        else:
            if name == "objects": 
                coco_to_model[coco_id] = 0 # unlabeled
            elif name == "construction-curb":
                coco_to_model[coco_id] = label2id.get("flat-curb", 0)
            else:
                coco_to_model[coco_id] = 0
                
    # Find images with annotations
    annotated_images = []
    for img in data['images']:
        sample_anns = [ann for ann in data['annotations'] if ann['image_id'] == img['id']]
        if len(sample_anns) > 0:
            annotated_images.append((img, sample_anns))
            if len(annotated_images) >= 3:
                break
                
    if not annotated_images:
        print("NO ANNOTATIONS FOUND!")
        return

    for img, sample_anns in annotated_images:
        print(f"\nVerifying image: {img['file_name']} ({len(sample_anns)} annotations)")
        
        mask = np.zeros((img['height'], img['width']), dtype=np.uint8)
        for ann in sample_anns:
            cat_id = ann['category_id']
            model_id = coco_to_model.get(cat_id, 0)
            
            if 'segmentation' in ann:
                # Handle RLE or Polygon
                m = mask_utils.decode(ann['segmentation'])
                mask[m > 0] = model_id
                    
        unique_labels = np.unique(mask)
        print(f"Unique labels in mask: {unique_labels}")
        for l in unique_labels:
            print(f"  ID {l}: {id2label.get(str(l), 'unknown')}")

if __name__ == "__main__":
    verify()
