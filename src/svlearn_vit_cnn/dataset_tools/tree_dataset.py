#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

import pandas as pd
from PIL import Image

# torch
import torch
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform

#  -------------------------------------------------------------------------------------------------


class TreeDataset(Dataset):
    def __init__(self, image_df: pd.DataFrame, transform: Transform):
        # list of paths to image files
        self.X_paths = image_df["image_path"].to_list()

        # list of corresponding labels
        self.labels = image_df["label"].to_list()

        # transform to apply to the image
        self.transform = transform

    def __len__(self):
        return len(self.X_paths)

    def __getitem__(self, idx):
        sample_path = self.X_paths[idx]
        label = self.labels[idx]

        image: Image = Image.open(sample_path).convert("RGB")

        return self.transform(image), torch.tensor(label, dtype=torch.long)


#  -------------------------------------------------------------------------------------------------
