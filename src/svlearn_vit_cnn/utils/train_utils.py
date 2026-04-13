# huggingface
from transformers import EvalPrediction
import evaluate
from datasets import Dataset

import numpy as np
from scipy.special import softmax
from PIL import Image
from typing import Tuple, Callable
from torchvision import transforms

import torch

# svlearn
from svlearn_vit_cnn.dataset_tools.preprocess import Preprocessor
from svlearn_vit_cnn.utils.visualization_utils import plot_roc_curve

# sklearn
from sklearn.preprocessing import LabelEncoder

#  -------------------------------------------------------------------------------------------------

def get_device():
    """
    Detects and returns the best available device (CUDA, MPS, or CPU).
    
    Returns:
        torch.device: The device to use for training
        bool: Whether to use fp16 (only for CUDA)
        bool: Whether to use bf16 (only for MPS)
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return device, True, False  # fp16=True for CUDA
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon (MPS) device")
        return device, False, True  # bf16=True for MPS
    else:
        device = torch.device("cpu")
        print("Using CPU device")
        return device, False, False  # No mixed precision for CPU

#  -------------------------------------------------------------------------------------------------

def print_trainable_parameters(model):
    """prints the trainable parameters of a model
    Args:
        model (_type_): any huggingface transformer model
    """

    # Count total and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Calculate the percentage
    trainable_percentage = (trainable_params / total_params) * 100
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")
    print(f"Percentage of trainable parameters: {trainable_percentage:.2f}%")

#  -------------------------------------------------------------------------------------------------

def compute_metrics_old_gunshot(eval_pred: EvalPrediction, results_dir: str):
    """does the computation necessary during evaluation

    Args:
        eval_pred (EvalPrediction): evaluation predictions
        results_dir (str): directory to save ROC curve plot

    Returns:
        dict: dictionary with accuracy, precision, recall, and f1 metrics
    """
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load('recall')
    f1_metric = evaluate.load('f1')
    
    predictions = np.argmax(eval_pred.predictions, axis=1)
    probs = softmax(eval_pred.predictions, axis=1)[:, 1]
    accuracy = accuracy_metric.compute(predictions=predictions, references=eval_pred.label_ids)
    precision = precision_metric.compute(predictions=predictions, references=eval_pred.label_ids)
    recall = recall_metric.compute(predictions=predictions, references=eval_pred.label_ids)
    f1 = f1_metric.compute(predictions=predictions, references=eval_pred.label_ids)

    plot_roc_curve(eval_pred.label_ids, probs, f"{results_dir}/roc.png", False)

    return {'accuracy': accuracy['accuracy'], 
            'precision': precision['precision'], 
            'recall': recall['recall'], 
            'f1': f1['f1']}

#  -------------------------------------------------------------------------------------------------






def compute_metrics(eval_pred: EvalPrediction, results_dir: str):
    """does the computation necessary during evaluation for MULTICLASS"""
    
    accuracy_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load('recall')
    f1_metric = evaluate.load('f1')
    
    # 1. Get the predicted class index (highest probability)
    predictions = np.argmax(eval_pred.predictions, axis=1)
    
    # 2. Extract references
    references = eval_pred.label_ids

    # 3. Compute Metrics with 'macro' averaging for multiclass
    accuracy = accuracy_metric.compute(predictions=predictions, references=references)
    
    # We add average="macro" here so it calculates scores for Me, Daughter, and Son
    precision = precision_metric.compute(predictions=predictions, references=references, average="macro")
    recall = recall_metric.compute(predictions=predictions, references=references, average="macro")
    f1 = f1_metric.compute(predictions=predictions, references=references, average="macro")

    # 4. ROC Curve logic (Optional for Multiclass)
    # Note: Traditional ROC is binary. For 3 classes, it's complex. 
    # We'll just pass the first class probabilities to avoid crashing the plot function.
    probs = softmax(eval_pred.predictions, axis=1)[:, 0] 
    try:
        plot_roc_curve(references, probs, f"{results_dir}/roc.png", False)
    except:
        print("Skipping ROC plot: Multiclass ROC requires specific handling.")

    return {
        'accuracy': accuracy['accuracy'], 
        'precision': precision['precision'], 
        'recall': recall['recall'], 
        'f1': f1['f1']
    }
    #  -------------------------------------------------------------------------------------------------

def collate_fn(batch):
    """process a list of samples into a torch batch

    Args:
        batch (_type_): a dataset batch with list of dicts

    Returns:
        _type_: a torch batch
    """
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }

#  -------------------------------------------------------------------------------------------------

def make_train_transform(processor):
    """Creates a train transform function that processes images at runtime with augmentations

    Args:
        processor: HuggingFace image processor (ViTImageProcessor or AutoImageProcessor)

    Returns:
        Callable: train transform function
    """
    def train_transform(example_batch):
        """processes the image at runtime for training (no augmentations for spectrograms)
        """
        batch = [Image.open(sample_path).convert("RGB") for sample_path in example_batch['image_path']]

        # Take a list of PIL images and turn them to pixel values
        inputs = processor(batch, return_tensors='pt')

        inputs['labels'] = example_batch['label']
        return inputs
    
    return train_transform

def make_test_transform(processor):
    """Creates a test transform function that processes images at runtime without augmentations

    Args:
        processor: HuggingFace image processor (ViTImageProcessor or AutoImageProcessor)

    Returns:
        Callable: test transform function
    """
    def test_transform(example_batch):
        """processes the image at runtime for evaluation/testing

        Args:
            example_batch: a batch of the dataset

        Returns:
            Dict: a dictionary with keys (pixel_values, labels)
        """
        batch = [Image.open(sample_path).convert("RGB") for sample_path in example_batch['image_path']]

        # Take a list of PIL images and turn them to pixel values
        inputs = processor(batch, return_tensors='pt')

        inputs['labels'] = example_batch['label']
        return inputs
    
    return test_transform

#  -------------------------------------------------------------------------------------------------

def prepare_datasets(raw_dir: str, processed_dir: str, train_transform: Callable, test_transform: Callable, save_json: bool = False) -> Tuple[Dataset, Dataset, LabelEncoder]:
    """prepares the train and test datasets for image classification models

    Args:
        raw_dir (str): path to raw training data directory
        processed_dir (str): path to processed data directory
        train_transform (Callable): transform function for training dataset
        test_transform (Callable): transform function for test/validation dataset
        save_json (bool, optional): save the train and validation dataframes. Defaults to False.

    Returns:
        Tuple[Dataset, Dataset, LabelEncoder]: train_dataset, test_dataset, label_encoder
    """
    preprocessor = Preprocessor()

    # load train_df
    train_df, val_df, label_encoder = preprocessor.preprocess(raw_dir, processed_dir)

    # create a dataset suitable for training
    train_dataset = Dataset.from_pandas(train_df).with_transform(train_transform)
    val_dataset = Dataset.from_pandas(val_df).with_transform(test_transform)

    return train_dataset, val_dataset, label_encoder

#  -------------------------------------------------------------------------------------------------