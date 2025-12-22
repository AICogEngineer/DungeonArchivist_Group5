# Trains the Keras model on the known labeled data

"""
PHASE 1: THE VISION MODEL (DATASET A)

Purpose:
- Teach the model what different game assets look like using labeled data (Dataset A)
- Train a CNN classifier on 32 by 32 images
- Extract compact embedding vectors from the trained model
- Store embeddings in a vector database (ChromaDB) for similarity search

This file does NOT yet:
- Touch Dataset B
- Move or delete files
- Perform predictions on unknown data

"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Input, Model

# Getting dataset: Open File Explorer, open the dataset, right click the dataset name, select Copy address as text, and paste that below
datasetA = "C:\Harsh\Jobs\Revature\ProjectCode\GitHub\AI_ML_VibeCoding\DungeonArchivist_Group5\DungeonCrawlStoneSoupFull_Dataset"

IMG_SIZE = (32, 32)    # Image width & height (pixels)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.003  # Step size for gradient updates
EMBEDDING_DIM = 64     # Size of the learned feature vector
VALIDATION_SPLIT = 0.2                # 80/20 train/validation split
SEED = 42

# Each subfolder name becomes a class label
class_names = sorted([
    d for d in os.listdir(datasetA)
    if os.path.isdir(os.path.join(datasetA, d))
])

num_classes = len(class_names)
class_to_index = {name: idx for idx, name in enumerate(class_names)}
print("Detected classes:", class_names)

