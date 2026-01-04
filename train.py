# Trains the Keras model on the known labeled data

"""
PHASE 1: THE VISION MODEL (DATASET A)

Purpose:
- Train a CNN on labeled 32 by 32 game assets (Dataset A)
- Extract compact embedding vectors from the trained model
- Store normalized embeddings in a vector database (ChromaDB) for cosine similarity search
"""

import os
import datetime
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

# Class label map from dataset A's hierarchy
class_map = {
    "dungeon": [
        "altars","grass","sigils","gateways","shops","statues","traps",
        "trees","vaults","abyss","banners","torches","water"
    ],
    "effect": ["effect"],
    "emissary": ["emissaries"],
    "gui": [
        "abilities","commands","invocations","skills","air","components",
        "conjuration","disciplines","divination","earth","enchantment",
        "fire","ice","monster","necromancy","poison","summoning",
        "translocation","transmutation","startup","tabs"
    ],
    "item": [
        "amulet","armor","book","food","gold","runes","ring","rod",
        "scroll","staff","wand","weapon","artefact","ranged"
    ],
    "misc": ["blood","brands","numbers"],
    "monster": [
        "aberration","abyss","amorphous","animals","aquatic","demons",
        "demonspawn","draconic","dragons","eyes","fungi_plants","holy",
        "nonliving","panlord","spriggan","statues","tentacles",
        "undead","unique","vault"
    ],
    "player": [
        "barding","base","beard","body","boots","cloak","draconic",
        "enchantment","felids","gloves","hair","halo","hand_left",
        "hand_right","heads","legs","mutations","transform"
    ]
}

def map_to_parent(folder_name):
    for parent, children in class_map.items():
        if folder_name in children:
            return parent
    return None

# Collects images
image_paths = [] # Stores full paths to all images
hierarchy_labels = [] # Stores parent class labels derived from folder structure

# Walk through Dataset A
for root, _, files in os.walk(datasetA):
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            rel = os.path.relpath(root, datasetA)
            top = rel.split(os.sep)[0] if rel != "." else None
            parent_class = map_to_parent(top)
            if parent_class:
                image_paths.append(os.path.join(root, file)) # Stores image path
                hierarchy_labels.append(parent_class)

class_names = sorted(set(hierarchy_labels)) # Unique parent-level class names
class_to_index = {name: i for i, name in enumerate(class_names)} # Map each hierarchical class to a numeric index
labels = np.array([class_to_index[l] for l in hierarchy_labels]) # Convert hierarchy labels to numeric labels
image_paths = np.array(image_paths) # Convert image paths to NumPy array for indexing

print(f"Detected {len(class_names)} class-level labels")

# Shuffles the data once using a fixed seed so results are repeatable
SEED = 42
rng = np.random.default_rng(SEED)
indices = rng.permutation(len(image_paths)) # Generates the shuffled indices
image_paths = image_paths[indices]
labels = labels[indices]

# Train / validation split
split_index = int(len(image_paths) * (1 - VALIDATION_SPLIT)) # Finds the split index
train_paths = image_paths[:split_index]
val_paths = image_paths[split_index:]
train_labels = labels[:split_index]
val_labels = labels[split_index:]

def load_image(path, label):
    image_bytes = tf.io.read_file(path) # Reads the image
    image = tf.image.decode_image(image_bytes, channels = 4, expand_animations = False) # Decodes image as RGBA and ignores animations like GIFs
    image = tf.image.resize(image, IMG_SIZE) # Images resized to 32 by 32

    rgb = tf.cast(image[..., :3], tf.float32) / 255.0 # Normalizes RGB pixel values to [0, 1]
    alpha = tf.cast(image[..., 3:4], tf.float32) / 255.0 # Alpha channel in [0, 1]

    # Composites transparent pixels onto a white background so sprites are consistent
    bg = tf.ones_like(rgb)
    image = rgb * alpha + bg * (1.0 - alpha)

    return image, label # Label passed through for training

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
        x, y = [], []

        for i in batch_indices:
            img, label = load_image(self.paths[i], self.labels[i])
            x.append(img)
            y.append(label)

        return np.array(x), np.array(y) # Convert lists to NumPy arrays for keras

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices) # Shuffle indices after each epoch to improve training

# Instantiate generators for training and validation
train_gen = ImageSequence(train_paths, train_labels)
val_gen = ImageSequence(val_paths, val_labels, shuffle = False) # Validation data is not shuffled

inputs = layers.Input(shape = (32, 32, 3)) # 32 by 32 RGB images

x = layers.RandomFlip("horizontal")(inputs)
x = layers.RandomRotation(0.3)(x) # so it is trained on images that are rotated 
x = layers.RandomZoom(0.2)(x) # so it is trained on images that are zoomed in

x = layers.Conv2D(32, 3, padding = "same")(x) # Conv layer: 32 filters, 3x3 kernel
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D()(x)

x = layers.Conv2D(64, 3, padding = "same")(x) # 64 filters, 3x3 kernel
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D()(x)

x = layers.GlobalAveragePooling2D()(x)
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

# Create unique log directory with timestamp
log_dir = os.path.join("logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
os.makedirs(log_dir, exist_ok = True)

tensorboard_callback = callbacks.TensorBoard(
    log_dir = log_dir,
    histogram_freq = 1,
    write_graph = True,
    write_images = False,
    update_freq = "epoch",
    profile_batch = 0
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
    callbacks = [early_stop, tensorboard_callback],
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
    metadata = {"hnsw:space": "cosine"} # cosine similarity
)

for path, label_idx in zip(image_paths, labels):
    img, _ = load_image(path, 0)
    img = np.expand_dims(img, axis = 0) # Expands dims because Keras models expect batch input

    vector = embedding_model.predict(img, verbose = 0)[0].astype(np.float32) # Generates the embedding vector for the image

    # Normalizes embedding for correct cosine similarity
    vector = vector / np.linalg.norm(vector)  # Makes sure comparisons are fair distances
                    # np.linalg.norm(vector) computes the length (magnitude) of the vector, in this case the above line of code scales the vector so its length becomes exactly 1 for vector normalization.

    # Create a stable, repeatable ID for the image
        # Stable ID ensures re-runs overwrite instead of duplicating
    rel_path = os.path.relpath(path, datasetA) # The full relative image path
    stable_id = hashlib.md5(rel_path.encode()).hexdigest() # Using the relative path ensures the same image gets the same ID across multiple runs, preventing duplicate entries

    collection.upsert( # Store / update the image embedding in ChromaDB
        embeddings = [vector.tolist()], # Numerical vector for similarity search
        ids = [stable_id],
        metadatas = [{
            "label": class_names[label_idx], # The parent class label
            "path": rel_path # The relative path to the image file
        }]
    )

# Just so there is some kind of results printed at the end of training
final_val_acc = max(model.history.history["val_accuracy"])
print(f"\nBest validation accuracy: {final_val_acc:.4f}")