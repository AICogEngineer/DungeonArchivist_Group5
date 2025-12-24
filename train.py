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
from keras import layers, models, optimizers
import chromadb

# Getting dataset: Open File Explorer, open the dataset, right click the dataset name, select Copy address as text, and paste that below
datasetA = "C:\Harsh\Jobs\Revature\ProjectCode\GitHub\AI_ML_VibeCoding\DungeonArchivist_Group5\DungeonCrawlStoneSoupFull_Dataset"

IMG_SIZE = (32, 32)     # Image width & height (pixels)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.003   # Step size for gradient updates
EMBEDDING_DIM = 64      # Size of the learned feature vector
VALIDATION_SPLIT = 0.2  # 80/20 train/validation split

# Each subfolder name becomes a class label
class_names = sorted([
    d for d in os.listdir(datasetA)
    if os.path.isdir(os.path.join(datasetA, d))
])

num_classes = len(class_names) 
class_to_index = {name: idx for idx, name in enumerate(class_names)}
print("Detected classes:", class_names)

# Empty but will be filled in the for loop below
image_paths = []
labels = []

# Collects image file paths and their numeric class labels, instead of loading images immediately, to  load them efficiently later
for class_name in class_names:
    class_dir = os.path.join(datasetA, class_name)
    for file_name in os.listdir(class_dir): # Iterate over files inside the class folder
        if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            image_paths.append(os.path.join(class_dir, file_name)) # Saves full image path
            labels.append(class_to_index[class_name]) # Saves class index
        else:
            print(f"Skipped the incorrect file type: {file_name}") 
            print("Correct file types include: .png, .jpg or .jpeg")

# Convert lists to NumPy arrays for indexing
image_paths = np.array(image_paths)
labels = np.array(labels)

# Shuffles the data once using a fixed seed so results are repeatable
SEED = 42
rng = np.random.default_rng(SEED)
indices = rng.permutation(len(image_paths)) # Generate shuffled indices
image_paths = image_paths[indices]
labels = labels[indices]

# Train / validation split
split_index = int(len(image_paths) * (1 - VALIDATION_SPLIT))
train_paths, val_paths = image_paths[:split_index], image_paths[split_index:]
train_labels, val_labels = labels[:split_index], labels[split_index:]

def load_image(path, label):
    image = tf.io.read_file(path) # Reads the image
    image = tf.image.decode_image(image, channels = 3, expand_animations = False) # Decodes image as RGB
    image = tf.image.resize(image, IMG_SIZE) # Images resized to 32 by 32
    image = image / 255.0  # Normalize pixel values to [0, 1]

    return image, label

class ImageSequence(tf.keras.utils.Sequence):
    def __init__(self, paths, labels, batch_size = BATCH_SIZE, shuffle = True):
        self.paths = paths # List of file paths to images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle # Determines whether to shuffle at end of epoch
        self.indices = np.arange(len(self.paths)) # Array of indices for shuffling
        if shuffle:
            np.random.shuffle(self.indices) # First shuffle

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size)) # Returns number of batches per epoch

    def __getitem__(self, idx): # Generate one batch of data
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size] # Select indices for this batch
        batch_x, batch_y = [], []

        for i in batch_indices:
            img, label = load_image(self.paths[i], self.labels[i])
            batch_x.append(img)
            batch_y.append(label)

        return np.array(batch_x), np.array(batch_y) # Convert lists to NumPy arrays for keras

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices) # Shuffle indices after each epoch to improve training

# Instantiate generators for training and validation
train_gen = ImageSequence(train_paths, train_labels)
val_gen = ImageSequence(val_paths, val_labels, shuffle = False) # Validation data is not shuffled





class ChromaDBHandler:
    def __init__(self, artifacts_dir, chroma_path, collection_name):
        self.artifacts_dir = artifacts_dir
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        
        self.client = chromadb.PersistentClient(path = self.chroma_path)
        self.collection = self.client.get_or_create_collection(
            name = self.collection_name,
            metadata = {"hnsw:space": "cosine"},
        )

chroma_handler = ChromaDBHandler(
    artifacts_dir = datasetA,
    chroma_path = "./chroma_db",
    collection_name = "datasetA_embeddings"
)
collection = chroma_handler.collection

#Build CNN and embedding layer
inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="image")

x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(inputs)
x = layers.Activation("relu", name="relu1")(x)
x = layers.MaxPooling2D((2, 2), name="pool1")(x)

x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
x = layers.Activation("relu", name="relu2")(x)
x = layers.MaxPooling2D((2, 2), name="pool2")(x)

x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
x = layers.Activation("relu", name="relu3")(x)

x = layers.Flatten(name="flatten")(x)
embedding = layers.Dense(EMBEDDING_DIM, activation=None, name="embedding")(x)
embedding_norm = layers.LayerNormalization(name="embedding_norm")(embedding)

x = layers.Dropout(0.2, name="dropout")(embedding_norm)
outputs = layers.Dense(num_classes, activation="softmax", name="class_probs")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="dungeon_cnn")
model.summary()