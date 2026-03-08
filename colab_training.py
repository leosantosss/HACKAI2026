# Vizzion Navigation Master: High-Precision Colab Training
# This script is optimized for Google Colab (T4 GPU)

# 1. INSTALL DEPENDENCIES (Run this in the first cell)
# !pip install -q transformers datasets evaluate pycocotools roboflow accelerate

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
from roboflow import Roboflow

# --- CONFIG ---
# Replace with your Roboflow API Key or I can help you find it
ROBOFLOW_API_KEY = "YOUR_API_KEY_HERE" 
PROJECT_ID = "vizzion"
VERSION = 1

ID2LABEL_FILE = "id2label.json"
MODEL_CHECKPOINT = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024" 
OUTPUT_DIR = "vizzion-navigation-master"

# 2. DOWNLOAD DATASET
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace().project(PROJECT_ID)
dataset = project.version(VERSION).download("coco")
DATASET_ROOT = dataset.location

# 3. SETUP LABELS
# Paste the content of your local id2label.json here
id2label_data = {"0": "unlabeled", "1": "flat-road", "2": "flat-sidewalk", "3": "flat-crosswalk", "4": "flat-cyclinglane", "5": "flat-parkingdriveway", "6": "flat-railtrack", "7": "flat-curb", "8": "human-person", "9": "human-rider", "10": "vehicle-car", "11": "vehicle-truck", "12": "vehicle-bus", "13": "vehicle-tramtrain", "14": "vehicle-motorcycle", "15": "vehicle-bicycle", "16": "vehicle-caravan", "17": "vehicle-cartrailer", "18": "construction-building", "19": "construction-door", "20": "construction-wall", "21": "construction-fenceguardrail", "22": "construction-bridge", "23": "construction-tunnel", "24": "construction-stairs", "25": "object-pole", "26": "object-trafficsign", "27": "object-trafficlight", "28": "nature-vegetation", "29": "nature-terrain", "30": "sky", "31": "void-ground", "32": "void-dynamic", "33": "void-static", "34": "void-unclear"}
id2label = {int(k): v for k, v in id2label_data.items()}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

processor = SegformerImageProcessor.from_pretrained(MODEL_CHECKPOINT)
jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

def get_coco_mapping(coco_categories, label2id):
    mapping = {}
    for cat in coco_categories:
        name = cat['name']
        cid = cat['id']
        if name in label2id:
            mapping[cid] = label2id[name]
        elif name == "construction-curb":
            mapping[cid] = label2id.get("flat-curb", 0)
        else:
            mapping[cid] = 0
    return mapping

def load_coco_as_dict(split):
    coco_json = os.path.join(DATASET_ROOT, split, "_annotations.coco.json")
    img_dir = os.path.join(DATASET_ROOT, split)
    with open(coco_json) as f:
        data = json.load(f)
    mapping = get_coco_mapping(data['categories'], label2id)
    img_id_to_anns = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        id_list = img_id_to_anns.get(img_id, [])
        id_list.append(ann)
        img_id_to_anns[img_id] = id_list
    dataset_list = []
    for img_info in data['images']:
        img_id = img_info['id']
        img_path = os.path.join(img_dir, img_info['file_name'])
        height, width = img_info['height'], img_info['width']
        mask = np.zeros((height, width), dtype=np.uint8)
        anns = img_id_to_anns.get(img_id, [])
        for ann in anns:
            mid = mapping.get(ann['category_id'], 0)
            seg = ann['segmentation']
            if isinstance(seg, list):
                rle = mask_utils.frPyObjects(seg, height, width)
                m = mask_utils.decode(rle)
                if len(m.shape) > 2: m = np.max(m, axis=2)
                mask[m > 0] = mid
            else:
                m = mask_utils.decode(seg)
                mask[m > 0] = mid
        dataset_list.append({"pixel_values": img_path, "label": Image.fromarray(mask)})
    return dataset_list

def train_transforms(example_batch):
    images = []
    labels = []
    for x, y in zip(example_batch['pixel_values'], example_batch['label']):
        img = Image.open(x).convert("RGB")
        lab = y
        if np.random.rand() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            lab = lab.transpose(Image.FLIP_LEFT_RIGHT)
        img = jitter(img)
        images.append(img)
        labels.append(lab)
    inputs = processor(images, labels, return_tensors="pt")
    return inputs

def val_transforms(example_batch):
    images = [Image.open(x).convert("RGB") for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    return processor(images, labels, return_tensors="pt")

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        weights = torch.ones(num_labels).to(logits.device)
        weights[7] = 8.0   # Curb
        weights[24] = 8.0  # Stairs
        weights[8] = 4.0   # Person
        loss_fct = nn.CrossEntropyLoss(weight=weights, ignore_index=255)
        upsampled_logits = nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
        loss = loss_fct(upsampled_logits, labels)
        return (loss, outputs) if return_outputs else loss

train_ds = Dataset.from_list(load_coco_as_dict("train"))
valid_ds = Dataset.from_list(load_coco_as_dict("valid"))
train_ds.set_transform(train_transforms)
valid_ds.set_transform(val_transforms)

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_CHECKPOINT, num_labels=num_labels, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
).to("cuda") # USE GPU

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=1e-4,
    num_train_epochs=15, 
    per_device_train_batch_size=4, # INCREASE BATCH SIZE ON GPU
    per_device_eval_batch_size=4,
    save_total_limit=2,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    remove_unused_columns=False,
    report_to="none",
    fp16=True # ENABLE MIXED PRECISION FOR EXTRA SPEED
)

trainer = WeightedTrainer(
    model=model, args=training_args, train_dataset=train_ds, eval_dataset=valid_ds
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)


