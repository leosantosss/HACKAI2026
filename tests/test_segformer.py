"""
test_segformer.py — Vizzion SegFormer Sidewalk Test
Reads images from the segments/sidewalk-semantic parquet dataset,
runs chainyo/segformer-sidewalk, and shows:
  LEFT:  Original image
  RIGHT: Predicted segmentation (color-coded)
Press any key for next image, Q to quit.
"""
import os
import io
import json
import cv2
import numpy as np
import torch
import pandas as pd
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

MODEL_NAME    = "nvidia/segformer-b1-finetuned-cityscapes-1024-1024"
PARQUET_FILE  = "train-00000-of-00001.parquet"
ID2LABEL_FILE = "id2label.json"

# 1. Dataset Labels (for Ground Truth)
with open(ID2LABEL_FILE) as f:
    ds_id2label = {int(k): v for k, v in json.load(f).items()}

# 2. Load Model & Get Model Labels
print(f"\nLoading {MODEL_NAME}...")
processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
model.eval()
print("Model ready.")

model_id2label = model.config.id2label
num_model_classes = len(model_id2label)

# 3. Create Color Palettes
def get_palette(id2label_dict):
    rng = np.random.default_rng(42)
    palette = (rng.integers(60, 230, size=(len(id2label_dict)+10, 3))).tolist()
    
    # Priority colors
    class_map = {
        "sidewalk": (244, 35, 232),  # Pink
        "road":     (128, 64, 128),  # Purple
        "curb":     (0, 165, 255),   # Orange
        "pole":     (192, 128, 128), # Grey-Blue
        "stairs":   (0, 200, 200),   # Teal
        "wall":     (102, 0, 0),     # Dark Red
        "person":   (60, 20, 220),   # Red
        "car":      (0, 0, 142),     # Blue
    }
    
    for cls_id, label in id2label_dict.items():
        label = label.lower()
        for key, color in class_map.items():
            if key in label:
                palette[int(cls_id)] = list(color)
                break
    return palette

ds_palette = get_palette(ds_id2label)
model_palette = get_palette(model_id2label)

def colorize(mask, palette):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in range(len(palette)):
        color_mask[mask == cls_id] = palette[cls_id]
    return color_mask

# Load model
print(f"\nLoading {MODEL_NAME}...")
processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
model.eval()
print("Model ready.\n")

# Load dataset
print(f"Reading {PARQUET_FILE}...")
df = pd.read_parquet(PARQUET_FILE)
print(f"Dataset has {len(df)} rows. Columns: {list(df.columns)}\n")
print("Press any key to advance, Q to quit.")

for i, row in df.iterrows():
    # 1. Extract Image
    img_data = row.get('pixel_values') or row.get('image')
    if isinstance(img_data, dict):
        raw = img_data.get('bytes') or img_data.get('path')
        pil_img = Image.open(io.BytesIO(raw)).convert('RGB')
    else:
        pil_img = Image.fromarray(np.array(img_data)).convert('RGB')
    
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # 2. Extract Ground Truth Mask
    label_data = row.get('label') or row.get('labels')
    if isinstance(label_data, dict):
        raw_mask = label_data.get('bytes')
        gt_pil = Image.open(io.BytesIO(raw_mask))
        gt_mask = np.array(gt_pil).astype(np.uint8)
    else:
        gt_mask = np.array(label_data).astype(np.uint8)

    # 3. Model Prediction
    print(f"[{i+1}/{len(df)}] Segmenting...", end=" ", flush=True)
    with torch.no_grad():
        inputs = processor(images=pil_img, return_tensors="pt")
        outputs = model(**inputs)

    logits = outputs.logits
    upsampled = torch.nn.functional.interpolate(
        logits, size=(frame.shape[0], frame.shape[1]),
        mode='bilinear', align_corners=False
    )
    pred_mask = upsampled.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
    
    # 4. Colorize
    gt_overlay   = colorize(gt_mask,   ds_palette)
    pred_overlay = colorize(pred_mask, model_palette)

    # 5. Build 3-Pane View
    # Pane 1: Original | Pane 2: Ground Truth | Pane 3: Prediction
    gt_blend   = cv2.addWeighted(frame, 0.4, gt_overlay, 0.6, 0)
    pred_blend = cv2.addWeighted(frame, 0.4, pred_overlay, 0.6, 0)
    
    combined = np.hstack([frame, gt_blend, pred_blend])
    
    # Text Labels
    h, w = frame.shape[:2]
    cv2.putText(combined, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(combined, "Ground Truth", (w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(combined, "Prediction", (2*w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    print("Done.")
    
    # Save Results
    os.makedirs("test_results", exist_ok=True)
    save_path = f"test_results/comparison_{i+1}.png"
    cv2.imwrite(save_path, combined)
    print(f"Saved to {save_path}")
    
    if i >= 4: # Save first 5
        break



cv2.destroyAllWindows()
print("\nDone.")
