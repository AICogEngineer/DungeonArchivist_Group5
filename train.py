# Trains the Keras model on the known labeled data

"""
PHASE 1: THE VISION MODEL (DATASET A)

Purpose:
- Teach the model what different game assets look like using labeled data (Dataset A)
- Train a CNN classifier on 32 by 32 images
- Extract compact embedding vectors from the trained model
- Store embeddings in a vector database (ChromaDB) for similarity search

"""

import os
# Suppresses TensorFlow and libpng warnings that are not relevant to model training. These warnings come from the image metadata (iCCP profiles) and do not affect the model learning
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models, optimizers, callbacks
import chromadb
import hashlib

# If this doesn't work the to get the dataset: Open File Explorer, open the dataset, right click the dataset name, select Copy address as text, and paste that below
datasetA = os.environ.get(
    "DATASET_A_DIR", 
    "./DungeonCrawlStoneSoupFull_DatasetA"
    )

IMG_SIZE = (32, 32)     # Image width & height (pixels)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.003   # Step size for gradient updates
EMBEDDING_DIM = 64      # Size of the learned feature vector
VALIDATION_SPLIT = 0.2  # 80/20 train/validation split

# Collects images recursively
image_paths = [] # Stores full paths to all images
hierarchy_labels = [] # Stores full relative folder paths

# Walk through Dataset A recursively (level 1, 2, 3, ...)
for root, _, files in os.walk(datasetA):                     # CHANGED: recursive scan
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            image_paths.append(os.path.join(root, file))     # Store image path
            hierarchy_labels.append(
                os.path.relpath(root, datasetA)              # Store FULL hierarchy path
            )

class_names = sorted(set(hierarchy_labels)) # Create a sorted list of unique hierarchical class names
class_to_index = {name: i for i, name in enumerate(class_names)} # Map each hierarchical class to a numeric index
labels = np.array([class_to_index[l] for l in hierarchy_labels]) # Convert hierarchy labels to numeric labels
image_paths = np.array(image_paths) # Convert image paths to NumPy array for indexing

print(f"Detected {len(class_names)} hierarchical classes")

# Shuffles the data once using a fixed seed so results are repeatable
SEED = 42
rng = np.random.default_rng(SEED)
indices = rng.permutation(len(image_paths)) # Generates the shuffled indices
image_paths = image_paths[indices]
labels = labels[indices]

# Train / validation split
split_index = int(len(image_paths) * (1 - VALIDATION_SPLIT)) # Finds the split index
train_paths, val_paths = image_paths[:split_index], image_paths[split_index:]
train_labels, val_labels = labels[:split_index], labels[split_index:]

def load_image(path, label):
    image = tf.io.read_file(path) # Reads the image
    image = tf.image.decode_image(image, channels = 3, expand_animations = False) # Decodes image as RGB
    image = tf.image.resize(image, IMG_SIZE) # Images resized to 32 by 32
    image = image / 255.0  # Normalize pixel values to [0, 1]

    return image, label # Label unused here; required only for training pipeline

class ImageSequence(tf.keras.utils.Sequence):
    def __init__(self, paths, labels, shuffle = True, **kwargs):
        super().__init__(**kwargs)  # To avoid the PyDataset warning this is required by Keras

        self.paths = paths # List of file paths to images
        self.labels = labels
        self.shuffle = shuffle # Determines whether to shuffle at end of epoch
        self.indices = np.arange(len(paths)) # Array of indices for shuffling
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.paths) / BATCH_SIZE)) # Returns number of batches per epoch

    def __getitem__(self, idx): # Generate one batch of data
        batch_indices = self.indices[idx * BATCH_SIZE:(idx + 1) * BATCH_SIZE] # Select indices for this batch
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

inputs = layers.Input(shape = (32, 32, 3)) # 32 by 32 RGB images

x = layers.Conv2D(32, 3, padding="same")(inputs) # Conv layer: 32 filters, 3x3 kernel
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D()(x)
# If you see overfitting add dropout here
x = layers.Conv2D(64, 3, padding="same")(x) # 64 filters, 3x3 kernel
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D()(x)
# If you see overfitting add dropout here
x = layers.Flatten()(x) # Flattening 2D feature maps to 1D vector
x = layers.Dropout(0.3)(x)

embedding = layers.Dense(EMBEDDING_DIM, name = "embedding")(x) # Embedding vector
embedding = layers.Dropout(0.2)(embedding)
outputs = layers.Dense(len(class_names), activation = 'softmax')(embedding)
model = models.Model(inputs, outputs)

model.compile(
    optimizer = optimizers.Adam(learning_rate = LEARNING_RATE),
    loss = 'sparse_categorical_crossentropy', # Sparse labels (integers)
    metrics = ['accuracy']
)

# Early stopping to prevent overfitting
early_stop = callbacks.EarlyStopping(
    monitor = "val_loss",
    patience = 3, # Stops if the val loss doesn't improve for 3 epochs
    restore_best_weights = True # Reverts to best-performing model
)

model.fit( # Training the model
    train_gen,
    validation_data = val_gen,
    epochs = EPOCHS,
    callbacks = [early_stop],
    verbose = 1
)

embedding_model = models.Model(
    inputs = model.input,
    outputs = model.get_layer("embedding").output # Extracts the embedding layer
)

embedding_model.save("./embedding_model.keras") # Actually saves the model so we can use it in archivist.py

client = chromadb.PersistentClient("./chroma_db")

collection = client.get_or_create_collection(
    name = "datasetA_embeddings",
    metadata = {"hnsw:space": "cosine"}
)

for path, label_idx in zip(image_paths, labels):
    img, _ = load_image(path, 0)
    img = np.expand_dims(img, axis = 0) # Expands dims because Keras models expect batch input

    vector = embedding_model.predict(img, verbose = 0)[0].astype(np.float32) # Generates the embedding vector for the image

    # Create a stable, repeatable ID for the image
        # Stable ID ensures re-runs overwrite instead of duplicating
    rel_path = os.path.relpath(path, datasetA) # The full relative image path
    stable_id = hashlib.md5(rel_path.encode()).hexdigest() # Using the relative path ensures the same image gets the same ID across multiple runs, preventing duplicate entries

    collection.upsert( # Store / update the image embedding in ChromaDB
        embeddings = [vector.tolist()], # Numerical vector for similarity search
        ids = [stable_id],
        metadatas = [{
            "label": class_names[label_idx], # The full hierarchy label
            "path": rel_path # The relative path to the image file
        }]
    )

# Just so there is some kind of results printed at the end of training
final_val_acc = max(model.history.history["val_accuracy"])
print(f"\nBest validation accuracy: {final_val_acc:.4f}")