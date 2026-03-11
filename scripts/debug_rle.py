import json
from pycocotools import mask as mask_utils

def debug_rle_all():
    coco_path = "/Users/leosantos/Documents/HACK_AI/vizzion-1/train/_annotations.coco.json"
    with open(coco_path) as f:
        data = json.load(f)
    
    print(f"Total annotations: {len(data['annotations'])}")
    
    dict_count = 0
    list_count = 0
    other_count = 0
    
    for i, ann in enumerate(data['annotations']):
        seg = ann.get('segmentation')
        if isinstance(seg, dict):
            dict_count += 1
            # Try decoding
            try:
                mask_utils.decode(seg)
            except Exception as e:
                print(f"Failed to decode dict at index {i}: {e}")
                print(f"Seg content: {seg}")
                break
        elif isinstance(seg, list):
            list_count += 1
            # For polygons, we use mask_utils.frPyObjects or similar, or skip decode
            # pycocotools.mask.decode usually expects RLE
            pass
        else:
            other_count += 1
            
    print(f"Dict (RLE): {dict_count}")
    print(f"List (Polygon): {list_count}")
    print(f"Other: {other_count}")

if __name__ == "__main__":
    debug_rle_all()
