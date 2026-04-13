#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------

import os
from typing import Dict, List, Any, Tuple
import pandas as pd

from svlearn.common.utils import directory_readable, ensure_directory
from svlearn_vit_cnn import config
# sklearn
from sklearn.preprocessing import LabelEncoder

#  -------------------------------------------------------------------------------------------------


class Preprocessor:
    def __init__(self) -> None:
        self.label_encoder = LabelEncoder()

    def load_classes(self, row: Dict[str, str]) -> List[Dict[str, Any]]:
        """Loads the class sub-directory paths from the root data directory

        Args:
            row (Dict[str, str]): path to root directory

        Returns:
            List[Dict[str, Any]]: paths to class sub-directories
        """
        data_dir = row["path"]

        directory_readable(data_dir)
        data = []
        for class_dir in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_dir)
            if os.path.isdir(class_path):
                data.append({"label": class_dir, "path": class_path})

        return data

    #  -------------------------------------------------------------------------------------------------

    def load_image_paths(self, row: Dict[str, str]) -> List[Dict[str, Any]]:
        """Loads the image paths in the class directory

        Args:
            row (Dict[str, str]): a row containting path to class directory

        Returns:
            List[Dict[str, Any]]: a list of rows containing image paths
        """
        class_dir = row["path"]
        class_label = row["label"]
        data = []

        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            data.append({"label": class_label, "image_path": img_path})

        return data

    #  -------------------------------------------------------------------------------------------------

    def split_dataframe(self, df: pd.DataFrame, test_frac=0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """_summary_

        Args:
            df (pd.DataFrame): _description_
            test_frac (float, optional): _description_. Defaults to 0.2.

        Returns:
            Tuple[pd.DataFrame , pd.DataFrame]: _description_
        """
        train = df.sample(frac=(1 - test_frac), random_state=200)
        test = df.drop(train.index).sample(frac=1.0)

        return train, test

    #  -------------------------------------------------------------------------------------------------

    def encode_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encodes labels using by fitting a LabelEncoder and transforming the label column

        Args:
            df (pd.DataFrame): _description_

        Returns:
            pd.DataFrame: _description_
        """

        df["label"] = self.label_encoder.fit_transform(df["label"])
        return df

    #  -------------------------------------------------------------------------------------------------

    def preprocess(self, data_dir: str,  processed_dir: str = "", save_json: bool =True ,) -> Tuple[pd.DataFrame, pd.DataFrame, LabelEncoder]:
        """preprocess to get image paths

        Args:
            data_dir (str): root directory containing subfolders for each class of trees

        Returns:
            Tuple[pd.DataFrame , pd.DataFrame , LabelEncoder]: train_df , test_df , label_encoder
        """
        class_dirs = self.load_classes({"path": data_dir})
        image_data = []

        for dir in class_dirs:
            image_data.extend(self.load_image_paths(dir))

        image_df = pd.DataFrame.from_records(image_data)
        encoded_image_df = self.encode_labels(image_df)

        train_df, test_df = self.split_dataframe(encoded_image_df, 0.2)

        if save_json:
            train_df.to_json(f"{processed_dir}/train.json", orient='records', index=False)
            test_df.to_json(f"{processed_dir}/validation.json", orient='records', index=False)

        return train_df, test_df, self.label_encoder



#  -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    data_dir = config['tree-dataset']['path']
    raw_dir = data_dir + '/train'
    processed_dir = data_dir + "/preprocessed"
    ensure_directory(processed_dir)

    preprocessor = Preprocessor()
    train_df, test_df, label_encoder = preprocessor.preprocess(raw_dir, processed_dir, save_json=True)

    print(train_df.head())

#  -------------------------------------------------------------------------------------------------
