import torch
import numpy as np
from PIL import Image
import os
import json
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import matplotlib.pyplot as plt

def test_model_batch():
    model_path = "vizzion-navigation-master"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Is training complete?")
        return

    # Load processor and model
    processor = SegformerImageProcessor.from_pretrained(model_path)
    model = SegformerForSemanticSegmentation.from_pretrained(model_path)
    model.eval()

    # Load labels
    id2label = model.config.id2label
    
    # Pick 6 images from test split
    test_dir = "/Users/leosantos/Documents/HACK_AI/vizzion-1/test"
    images = [f for f in os.listdir(test_dir) if f.endswith(".jpg")]
    if not images:
        print("No images found in test dir.")
        return
    
    # Sort or pick distinct ones
    images = sorted(images)[:6]
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 20))
    axes = axes.flatten()

    for i, img_name in enumerate(images):
        sample_img_path = os.path.join(test_dir, img_name)
        image = Image.open(sample_img_path).convert("RGB")
        
        # Predict
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Rescale logits to original image size
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False
        )
        prediction = upsampled_logits.argmax(dim=1).squeeze().numpy()

        # Create overlay
        viz_mask = np.zeros((*prediction.shape, 3), dtype=np.uint8)
        
        # Define high-visibility colors for key navigation classes
        color_map = {
            "flat-sidewalk": [0, 255, 0],    # Bright Green
            "flat-curb": [255, 0, 0],        # Red
            "flat-road": [0, 0, 255],        # Blue
            "human-person": [255, 255, 0],   # Yellow
            "vehicle-car": [255, 165, 0],    # Orange
            "construction-stairs": [255, 0, 255], # Magenta
            "object-pole": [128, 128, 128],  # Grey
        }

        for label_id in np.unique(prediction):
            label_name = id2label.get(str(label_id), id2label.get(int(label_id), "unknown"))
            
            if label_name in color_map:
                viz_mask[prediction == label_id] = color_map[label_name]
            else:
                # Random semi-transparent color for background classes
                np.random.seed(int(label_id))
                viz_mask[prediction == label_id] = np.random.randint(50, 150, size=3)

        # Alpha blend (0.4 image + 0.6 mask for better visibility)
        blend = (np.array(image) * 0.4 + viz_mask * 0.6).astype(np.uint8)
        axes[i].imshow(blend)
        # Add legend or label
        found_names = [id2label.get(str(lid), id2label.get(int(lid), str(lid))) for lid in np.unique(prediction) if lid != 0]
        axes[i].set_title(f"Test {i+1}: {', '.join(found_names[:3])}")
        axes[i].axis("off")
    
    output_path = "batch_verification_fixed.png"
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Fixed batch verification image saved to {output_path}")

if __name__ == "__main__":
    test_model_batch()
