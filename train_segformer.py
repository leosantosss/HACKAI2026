"""
train_segformer.py — Vizzion Precision Training Script
Fine-tunes SegFormer on the expanded 600-image Roboflow COCO dataset.
Uses weighted loss and data augmentation to improve curb/stair precision.
"""
import io
import os
import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from datasets import Dataset
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer
)
import evaluate
from torchvision.transforms import ColorJitter

# --- CONFIG ---
USE_CPU_ONLY = True   # FORCED TRUE to fix the MPS 'view size' bug
ID2LABEL_FILE = "id2label.json"
MODEL_CHECKPOINT = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024" 
OUTPUT_DIR = "vizzion-navigation-master"
DATASET_ROOT = "/Users/leosantos/Documents/HACK_AI/vizzion-1"

if USE_CPU_ONLY:
    os.environ["CUDA_VISIBLE_DEVICES"] = "" 
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# 1. Load Model Labels
with open(ID2LABEL_FILE) as f:
    id2label = {int(k): v for k, v in json.load(f).items()}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

# 2. Preparation Functions
processor = SegformerImageProcessor.from_pretrained(MODEL_CHECKPOINT)
jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

def get_coco_mapping(coco_categories, label2id):
    """Maps COCO category IDs to Model label IDs."""
    mapping = {}
    for cat in coco_categories:
        name = cat['name']
        cid = cat['id']
        if name in label2id:
            mapping[cid] = label2id[name]
        elif name == "objects":
            mapping[cid] = 0
        elif name == "construction-curb":
            mapping[cid] = label2id.get("flat-curb", 0)
        else:
            mapping[cid] = 0
    return mapping

def load_coco_as_dict(split):
    """Loads COCO split and returns images/labels as PIL objects."""
    coco_json = os.path.join(DATASET_ROOT, split, "_annotations.coco.json")
    img_dir = os.path.join(DATASET_ROOT, split)
    
    with open(coco_json) as f:
        data = json.load(f)
        
    mapping = get_coco_mapping(data['categories'], label2id)
    
    # Group annotations by image_id
    img_id_to_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
        
    dataset_list = []
    print(f"Loading {split} split ({len(data['images'])} images)...")
    
    for img_info in data['images']:
        img_id = img_info['id']
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        # Create mask
        height, width = img_info['height'], img_info['width']
        mask = np.zeros((height, width), dtype=np.uint8)
        
        anns = img_id_to_anns.get(img_id, [])
        for ann in anns:
            mid = mapping.get(ann['category_id'], 0)
            seg = ann['segmentation']
            
            # Handle Polygon or RLE
            if isinstance(seg, list):
                # Polygon
                rle = mask_utils.frPyObjects(seg, height, width)
                m = mask_utils.decode(rle)
                if len(m.shape) > 2:
                    m = np.max(m, axis=2)
                mask[m > 0] = mid
            else:
                # RLE
                m = mask_utils.decode(seg)
                mask[m > 0] = mid
            
        dataset_list.append({
            "pixel_values": img_path,
            "label": Image.fromarray(mask)
        })
        
    return dataset_list

def train_transforms(example_batch):
    images = []
    labels = []
    for x, y in zip(example_batch['pixel_values'], example_batch['label']):
        img = Image.open(x).convert("RGB") if isinstance(x, str) else x
        lab = y
        
        # Random Flip
        if np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lab = lab.transpose(Image.FLIP_LEFT_RIGHT)
            
        # Color Jitter
        img = jitter(img)
        
        images.append(img)
        labels.append(lab)
    
    inputs = processor(images, labels, return_tensors="pt")
    
    # MPS/Stride Fix
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.contiguous()
            
    return inputs

def val_transforms(example_batch):
    images = [Image.open(x).convert("RGB") if isinstance(x, str) else x for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    inputs = processor(images, labels, return_tensors="pt")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.contiguous()
    return inputs

# 3. Custom Weighted Trainer
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        # Define weights: Prioritize curbs (7), stairs (24), and people (8)
        weights = torch.ones(num_labels).to(logits.device)
        weights[7] = 8.0   # Curb
        weights[24] = 8.0  # Stairs
        weights[8] = 4.0   # Person
        
        loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=255)
        
        # Upsample logits to match labels
        upsampled_logits = nn.functional.interpolate(
            logits, 
            size=labels.shape[-2:], 
            mode="bilinear", 
            align_corners=False
        )
        
        loss = loss_fct(upsampled_logits, labels)
        return (loss, outputs) if return_outputs else loss

# 4. Load Data
train_data = load_coco_as_dict("train")
valid_data = load_coco_as_dict("valid")

train_ds = Dataset.from_list(train_data)
valid_ds = Dataset.from_list(valid_data)

train_ds.set_transform(train_transforms)
valid_ds.set_transform(val_transforms)

# 5. Initialize Model
print(f"Initializing {MODEL_CHECKPOINT} for {num_labels} classes...")
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)
if USE_CPU_ONLY:
    model = model.to("cpu")

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=1e-4,                # Slightly higher for refinement
    num_train_epochs=15,               # EXTENDED for precision
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    save_total_limit=2,
    eval_strategy="epoch",             # Evaluate at end of each epoch
    save_strategy="epoch",
    logging_steps=10,
    remove_unused_columns=False,
    push_to_hub=False,
    report_to="none",
    dataloader_pin_memory=False,
    dataloader_num_workers=0,
    fp16=False,
    use_cpu=True
)

# 7. Metrics
metric = evaluate.load("mean_iou")

def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits = torch.from_numpy(logits)
        labels = torch.from_numpy(labels)
        
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        preds = upsampled_logits.argmax(dim=1)
        
        metrics = metric.compute(
            predictions=preds.numpy(),
            references=labels.numpy(),
            num_labels=num_labels,
            ignore_index=255,
            reduce_labels=False
        )
        
        return {k: v.item() if (isinstance(v, np.ndarray) and v.size == 1) or hasattr(v, 'item') and not isinstance(v, np.ndarray) else v.tolist() if isinstance(v, np.ndarray) else v for k, v in metrics.items() if not (isinstance(v, np.ndarray) and v.size > 1)}

# 8. Start Refined Training
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    compute_metrics=compute_metrics,
)

print("\nStarting Navigation Master PRECISION REFINEMENT...")
# Wipe previous checkpoints for a fresh start or resume? 
# Let's start fresh to ensure weighted loss takes effect from step 0
if os.path.exists(OUTPUT_DIR):
    import shutil
    print(f"Cleaning previous output directory {OUTPUT_DIR} for a fresh start...")
    shutil.rmtree(OUTPUT_DIR)

trainer.train()

print(f"\nRefinement Complete. Saving to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("Done.")
