#  -------------------------------------------------------------------------------------------------
#   Copyright (c) 2024.  SupportVectors AI Lab
#   This code is part of the training material, and therefore part of the intellectual property.
#   It may not be reused or shared without the explicit, written permission of SupportVectors.
#
#   Use is limited to the duration and purpose of the training at SupportVectors.
#
#   Author: SupportVectors AI Training Team
#  -------------------------------------------------------------------------------------------------


# IMPORTS
#  -------------------------------------------------------------------------------------------------
import matplotlib.pyplot as plt
from typing import List
from numpy.typing import ArrayLike

from svlearn.common.utils import ensure_directory

import torch
from torchvision.transforms import v2
from sklearn.metrics import roc_curve , auc

from torch.utils.data import Dataset
#  -------------------------------------------------------------------------------------------------


def visualize_classification_training_results(
    training_losses: List[float],
    evaluation_losses: List[float],
    training_accuracies: List[float],
    evaluation_accuracies: List[float],
    dir_path: str = ".",
    filename: str = "training_results",
) -> None:
    """Visualizes the training and evaluation results over epochs.

    Args:
        training_losses (list of float): List of training losses recorded after each epoch.
        evaluation_losses (list of float): List of evaluation losses recorded after each epoch.
        training_accuracies (list of float): List of training accuracies recorded after each epoch.
        evaluation_accuracies (list of float): List of evaluation accuracies recorded after each epoch.
        dir_path (str): Directory path to save the generated plot. Defaults to current direcotry.
        filename (str): Filename of the plot. Defaults to 'training_results'
    """
    epochs = range(1, len(training_losses) + 1)

    plt.figure(figsize=(14, 5))

    # Plot training and evaluation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, training_losses, label="Training Loss")
    plt.plot(epochs, evaluation_losses, label="Evaluation Loss")
    plt.title("Training and Evaluation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot training and evaluation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, training_accuracies, label="Training Accuracy")
    plt.plot(epochs, evaluation_accuracies, label="Evaluation Accuracy")
    plt.title("Training and Evaluation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    ensure_directory(dir_path)
    plt.savefig(f"{dir_path}/{filename}.png")
    plt.show(block=True)  # waits till the plot window is closed before proceeding to the next execution.


#  -------------------------------------------------------------------------------------------------


def visualize_regression_training_results(
    training_losses: List[float],
    evaluation_losses: List[float],
    dir_path: str = ".",
    filename: str = "training_results",
):
    """
    Visualizes the training and evaluation results over epochs.

    Args:
        training_losses (list of float): List of training losses recorded after each epoch.
        evaluation_losses (list of float): List of evaluation losses recorded after each epoch.
        dir_path (str): Directory path to save the generated plot. Defaults to current direcotry.
        filename (str): Filename of the plot. Defaults to 'training_results'
    """
    epochs = range(1, len(training_losses) + 1)

    plt.figure(figsize=(14, 5))

    # Plot training and evaluation loss
    plt.plot(epochs, training_losses, label="Training Loss")
    plt.plot(epochs, evaluation_losses, label="Evaluation Loss")
    plt.title("Training and Evaluation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    ensure_directory(dir_path)
    plt.savefig(f"{dir_path}/{filename}.png")
    plt.show(block=True)  # waits till the plot window is closed before proceeding to the next execution.


#  -------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    training_losses = [0.8, 0.6, 0.4, 0.3, 0.2]
    evaluation_losses = [0.9, 0.7, 0.5, 0.35, 0.25]
    training_accuracies = [0.7, 0.8, 0.85, 0.9, 0.92]
    evaluation_accuracies = [0.68, 0.75, 0.8, 0.85, 0.88]

    visualize_classification_training_results(
        training_losses, evaluation_losses, training_accuracies, evaluation_accuracies
    )


def denormalize(tensor, mean, std):
    """Reverses the normalization applied to a tensor.

    Args:
        tensor (torch.Tensor): Normalized tensor image.
        mean (list): Mean used for normalization.
        std (list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: Denormalized image tensor.
    """
    mean = torch.tensor(mean).reshape(3, 1, 1)  # Reshape for broadcasting
    std = torch.tensor(std).reshape(3, 1, 1)    # Reshape for broadcasting
    denorm_img = tensor * std + mean
    return denorm_img

def show_image_with_denormalization(dataset: Dataset , sample_index: int = 0 ) -> None:
    """Displays a normalized tensor image after denormalizing.

    Args:
        tensor_image (torch.Tensor): Normalized tensor image.
    """

     # Get the image and label
    image, label = dataset[sample_index]

    # Denormalize the image (use the same mean and std as used in transforms.Normalize)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    denorm_img = denormalize(image, mean, std)
    
    # Convert the tensor image back to PIL image for display
    image = v2.ToPILImage()(denorm_img)
    
    # Display the image
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.title(f"Sample Image (Label: {label})")
    plt.axis('off')  # Hide axis for better visualization
    plt.show()

def show_sample_image(dataset: Dataset , sample_index: int = 0 ) -> None:
    """Displays a random sample image from the dataset.

    Args:
        dataset (torch.utils.data.Dataset): The dataset object containing images.
    """
    
    # Get the image and label
    image, label = dataset[sample_index]
    
    # Convert the tensor image back to PIL image for display
    image = v2.ToPILImage()(image)
    
    # Display the image
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.title(f"Sample Image (Label: {label})")
    plt.axis('off')  # Hide axis for better visualization
    plt.show()

def plot_roc_curve(y_true: ArrayLike, y_probs: ArrayLike, 
                   filename: str = None, show: bool = True):
    """plots  the roc curve and optionally saves it to a file.
    A receiver operating characteristic (ROC) curve is a graph that shows 
    how well a binary classifier model performs at different threshold values.

    Args:
        y_true (ArrayLike): the actual reference value (to compare with model predictions) as a list or numpy array
        y_probs (ArrayLike): the model output as a list or numpy array
        filename (str, optional): filepath to save the plot. Defaults to None.
        show (bool, optional): display the plot before returning. Defaults to True.
    """
    fpr , tpr, thresholds = roc_curve(y_true , y_probs)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve for each class
    plt.figure(figsize=(5,5))

    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    if filename:
        plt.savefig(filename)

    if show:
        plt.show()

    plt.close()

