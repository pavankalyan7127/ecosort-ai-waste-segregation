r"""Inspect BaseDataset signature.
"""

import inspect
from ultralytics.data.base import BaseDataset

print("BaseDataset signature:")
print(inspect.signature(BaseDataset.__init__))
