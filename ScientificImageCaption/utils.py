import torch
from torch.utils.data import default_collate
from PIL import Image

def collate_batch(batch):
    sample = batch[0]
    if isinstance(sample[0], Image.Image):
        attribute_count = len(sample)
        list_of_attributes = [list() for i in range(attribute_count)]
        for sample in batch:
            for i, j in enumerate(sample):
                list_of_attributes[i].append(j)

        for i, attribute_list in enumerate(list_of_attributes):
            if(type(attribute_list) in {int, float, torch.Tensor}):
                list_of_attributes[i] = torch.stack(attribute_list)
        return list_of_attributes
    else:
        try:
            return default_collate(batch)
        except:
            print(batch)