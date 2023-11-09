# Importing libraries
import itertools
import os
import pathlib
import random
import shutil
import typing as t
import numpy.typing as npt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.image as mpimg
import tensorflow as tf
from sklearn import metrics
from sklearn.metrics import confusion_matrix

# Utility helper functions
def set_random_seeds(seed: int = 42) -> None:
    """
    Sets random seed for reproducibility

    Args:
        seed (int, optional): Value of the seed. Defaults to 42.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

def create_class_weight(label_count_dict: t.Dict[str, int]) -> t.Dict[int, float]:
    """
    Creates class weights, optimally used for imbalanced data

    Args:
        label_count_dict (t.Dict[str, int]): Dictionary in the format {`label` : `count`}

    Returns:
        t.Dict[int, float]: Class weights dictionary in the format {`label` : `weight`}
    """
    total_samples = np.sum(list(label_count_dict.values()))
    labels = label_count_dict.keys()
    class_weights = dict()

    for idx, label in enumerate(labels):
        class_weights[idx] = float(total_samples/(len(list(labels))*label_count_dict[label]))

    return class_weights

def show_random_samples(directory_path: t.Union[pathlib.Path, str], label: str, num_of_samples: int = 5) -> None:
    """
    Shows sample of images from a directory belonging to a class

    Args:
        directory_path (t.Union[pathlib.Path, str]): Path of the parent directory where `label` is located
        label (str): Class label of the image
        num_of_samples (int, optional): Number of samples to show. Defaults to 5.
    """
    files = os.listdir(os.path.join(directory_path, label))
    plt.figure(figsize=(10, 12))
    plt.subplots_adjust(wspace=2)
    for i in range(num_of_samples):
        plt.subplot(1, num_of_samples, i+1)
        sample = random.choice(files)
        sample_path = os.path.join(os.path.join(directory_path, label), sample)
        img = mpimg.imread(sample_path)
        imgplot = plt.imshow(img)
        plt.axis(False)
        plt.title(f'{label.lower().capitalize()}\n{img.shape}')
    plt.show()
    
def create_model_checkpoint(model_name: str, save_path: str="model_experiments") -> tf.keras.callbacks.ModelCheckpoint:
    """
    Returns a model checkpoint callback ro save the best model while training
    
    Args:
        model_name (str): Name of the model, functions creates a directory for the model of this name
        save_path (str, optional): Main directory to store all weights. Defaults to "model_experiments".
    
    Returns:
        tf.keras.callbacks.ModelCheckpoint: A callback to get the best weights configuration according to the training.
    """
    return tf.keras.callbacks.ModelCheckpoint(filepath=f'{os.path.join(save_path, model_name)}.h5', verbose=1, save_best_only=True, save_weights_only=False)

def create_early_stopping(patience: int = 3, restore_best_weights: bool = True) -> tf.keras.callbacks.EarlyStopping:
    """
    Returns a early stopping callback
    
    Args:
        patience (int, optional): Number of iterations to look for improvement. Defaults to 3.
        restore_best_weights (bool, optional): Restore best weights. Defaults to True.
    
    Returns:
        tf.keras.callbacks.EarlyStoppin: A callback to stop training if the validation loss does not decrease.
    """
    return tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=patience, restore_best_weights=restore_best_weights)

def plot_loss_curves(history: tf.keras.callbacks.History) -> None:
    """
    Shows separate loss curves for training and validation metrics.
    
    Args:
        history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
    """

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(len(history.history['loss']))
    
    # Plot loss
    plt.figure(figsize=(19, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.xticks(epochs)
    plt.legend(loc='upper right')
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.xticks(epochs)
    plt.legend(loc='upper left');
    
def get_metrics(y_true: npt.ArrayLike, y_pred: npt.ArrayLike, plot_confusion_matrix: bool = False) -> t.Dict[str, float]:
    """
    Generates classification metrics, optionally shows confusion matrix

    Args:
        y_true (npt.ArrayLike): True labels
        y_pred (npt.ArrayLike): Predicted labels
        plot_confusion_matrix (bool, optional): Shows confusion matrix. Defaults to False.

    Raises:
        ValueError: When shape of `y_test` and `y_pred` are not the same.

    Returns:
        t.Dict[str, float]: Classification metrics
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shape mismatch error!")

    accuracy = metrics.balanced_accuracy_score(y_true, y_pred)
    precision = metrics.precision_score(y_true, y_pred)
    recall = metrics.recall_score(y_true, y_pred)
    f1_score = metrics.f1_score(y_true, y_pred, average='weighted')

    if plot_confusion_matrix:
        cnf_matrix = metrics.confusion_matrix(y_true, y_pred)
        sns.heatmap(cnf_matrix/np.sum(cnf_matrix), annot=True, fmt='.2%', cmap='Blues')
        plt.show()

    return {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-score': f1_score}

# Our function needs a different name to sklearn's plot_confusion_matrix
def make_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False): 
  """
  Makes a labelled confusion matrix comparing predictions and ground truth labels.
  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.
  
  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).
  
  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.
  
  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """  
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])
  
  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes), 
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)
  
  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Save the figure to the current working directory
  if savefig:
    fig.savefig("confusion_matrix.png")