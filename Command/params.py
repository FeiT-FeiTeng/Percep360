import os
import torch

def save_trainable_params(obj, f, prefix=""):
    seen = set()  # Track processed objects to avoid repeated recursion.
    for attr_name in dir(obj):
        attr = getattr(obj, attr_name)

        # Skip unhashable container types.
        if isinstance(attr, (dict, list, set)):
            continue

        # Use object ids to avoid TypeError on unhashable objects.
        if id(attr) in seen:
            continue
        seen.add(id(attr))
        
        new_prefix = f"{prefix}.{attr_name}" if prefix else attr_name
        if callable(attr) or isinstance(attr, (int, float, str)):
            f.write(f"{new_prefix}: {attr}\n")
        else:
            save_trainable_params(attr, f, prefix=new_prefix)
