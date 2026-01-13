"""
Patch LeRobot library to fix torch.stack() issue with HF Dataset Columns
"""
import torch
from lerobot.common.datasets import lerobot_dataset


_original_query_hf_dataset = lerobot_dataset.LeRobotDataset._query_hf_dataset


def patched_query_hf_dataset(self, query_indices):
    """Patched _query_hf_dataset that converts Columns to lists before stacking"""
    result = {}
    for key, q_idx in query_indices.items():
        if key in self.meta.video_keys:
            continue

        selected = self.hf_dataset.select(q_idx)
        column_data = selected[key]

        if hasattr(column_data, '__iter__') and not isinstance(column_data, (str, bytes)):
            tensor_list = [torch.as_tensor(x) for x in column_data]
            result[key] = torch.stack(tensor_list)
        else:
            result[key] = torch.as_tensor(column_data)

    return result


lerobot_dataset.LeRobotDataset._query_hf_dataset = patched_query_hf_dataset
print("✓ LeRobot library patched successfully")
