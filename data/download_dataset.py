from roboflow import Roboflow
import os

# --- DOWNLOAD FROM ROBOFLOW ---
def download():
    rf = Roboflow(api_key="tIgjsEiAwC7jS6i0CXCe")
    project = rf.workspace("leos-workspace-pibe8").project("vizzion")
    version = project.version(1)
    
    print("Downloading COCO Segmentation dataset...")
    dataset = version.download("coco-segmentation")
    print(f"Dataset downloaded to: {dataset.location}")

if __name__ == "__main__":
    download()
