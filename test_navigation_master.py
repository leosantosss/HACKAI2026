import torch
import numpy as np
from PIL import Image
import os
import json
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import matplotlib.pyplot as plt

def test_model():
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
    
    # Pick a random image from valid split
    valid_dir = "/Users/leosantos/Documents/HACK_AI/vizzion-1/valid"
    images = [f for f in os.listdir(valid_dir) if f.endswith(".jpg")]
    if not images:
        print("No images found in valid dir.")
        return
        
    sample_img_path = os.path.join(valid_dir, images[0])
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

    # Create visualization
    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    
    # Simple color palette for visualization
    # We'll highlight Sidewalk (2) and Curb (7)
    viz_mask = np.zeros((*prediction.shape, 3), dtype=np.uint8)
    for label_id in np.unique(prediction):
        label_name = id2label.get(str(label_id), "unknown")
        # Color specific classes
        if label_id == 2: # flat-sidewalk
            viz_mask[prediction == label_id] = [0, 255, 0] # Green
        elif label_id == 7: # flat-curb
            viz_mask[prediction == label_id] = [255, 0, 0] # Red
        elif label_id == 1: # flat-road
            viz_mask[prediction == label_id] = [0, 0, 255] # Blue
        elif label_id == 8: # human-person
            viz_mask[prediction == label_id] = [255, 255, 0] # Yellow
        else:
            # Use random color based on ID
            np.random.seed(int(label_id))
            viz_mask[prediction == label_id] = np.random.randint(50, 200, size=3)

    ax[1].imshow(viz_mask)
    ax[1].set_title("Prediction (Green: Sidewalk, Red: Curb, Blue: Road)")
    ax[1].axis("off")
    
    output_path = "final_verification.png"
    plt.savefig(output_path)
    print(f"Verification image saved to {output_path}")

if __name__ == "__main__":
    test_model()
