r"""Detailed inspection of Ultralytics dataset resolution and label caching.
"""

import inspect
from pathlib import Path
from ultralytics.data.utils import img2label_paths
import ultralytics.data.dataset as ds

print("=== img2label_paths Source Code ===")
print(inspect.getsource(img2label_paths))

sa, sb = f"{Path.sep}images{Path.sep}", f"{Path.sep}labels{Path.sep}"
print(f"Replacement pattern: sa='{sa}', sb='{sb}'")

test_path = [r"D:\EcoSort-Resources\datasets\TACO\processed\images\000101.jpg"]
print("Resolved for test_path:", img2label_paths(test_path))

# Also test if images are in ecosort_yolo/images/train/000101.jpg
test_path_yolo = [r"D:\EcoSort-Resources\datasets\TACO\ecosort_yolo\images\train\000101.jpg"]
print("Resolved for test_path_yolo:", img2label_paths(test_path_yolo))
