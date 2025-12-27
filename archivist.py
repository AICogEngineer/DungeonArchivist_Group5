# The main engine. Scans a target folder, queries Vector DB, and moves files to sorted subfolders.

"""
ARCHIVIST.PY — PHASE 2: THE RESTORATION ENGINE

Purpose:
- Scan an unlabeled folder (Dataset B / "Data Swamp")
- Convert each image into an embedding using the trained vision model
- Query ChromaDB for nearest neighbors
- Decide a label using similarity + voting
- Move files into restored folders or a review pile
"""

import os
import shutil # to move files across folders
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models
import chromadb

# Getting dataset do below or if that doesn't work: Open File Explorer, open the dataset, right click the dataset name, select Copy address as text, and paste that below
datasetB = os.environ.get(
    "DATASET_B_DIR",
    "./ChaosData_DatasetB"
)

# Output folders
restored_dir = "./Restored_Archive" # Where classified images will be moved, if model is confident
review_pile = "./Review_Pile" # Where images go if the model isn't confident enough to label them

IMG_SIZE = (32, 32) # Must match the training image size in train.py
k = 5 # Number of nearest neighbors to check
threshold = 0.35 # Confidence cutoff (lower = stricter)

def load_image(path):
    image = tf.io.read_file(path)

    image = tf.image.decode_image( # Decodes the image bytes into a tensor (RGB image)
        image,
        channels = 3,
        expand_animations = False
    )

    image = tf.image.resize(image, IMG_SIZE) # Resizes the image to 32 by 32 so it matches the model input
    image = image / 255.0

    return image

class ChromaDBHandler:
    def __init__(self, chroma_path, collection_name):
        self.client = chromadb.PersistentClient(path = chroma_path) # Connects to the persistent vector database

        self.collection = self.client.get_or_create_collection( # Loads or creates the collection storing the Dataset A embeddings
            name = collection_name,
            metadata = {"hnsw:space": "cosine"}  # Cosine similarity for vectors
            )

class Archivist:
    def __init__(self, model_path, chroma_handler):
        self.embedding_model = models.load_model(model_path)
        self.collection = chroma_handler.collection # Gets the ChromaDB collection

        # Makes sure the output folders actually exist
        os.makedirs(restored_dir, exist_ok = True)
        os.makedirs(review_pile, exist_ok = True)

    def embed_image(self, image_path):
        img = load_image(image_path)
        img = tf.expand_dims(img, axis = 0) # Adds a batch dimension because Keras wants it
        vector = self.embedding_model.predict(img, verbose = 0)[0] # Generates the embedding vector

        return vector.astype(np.float32)

    def query_db(self, vector):
        return self.collection.query( # Asks ChromaDB for the closest stored embeddings
            query_embeddings = [vector.tolist()],
            n_results = k,
            include = ["metadatas", "distances"]
            )

    def decide_label(self, results):
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        if distances[0] > threshold: # If the closest match is still too far away, reject it
            return None  # Will go to review pile

        # Weighted voting, so the closer neighbors count more
        votes = {}
        for meta, dist in zip(metadatas, distances):
            label = meta["label"]
            weight = 1.0 / (dist + 1e-6)
            votes[label] = votes.get(label, 0.0) + weight

        return max(votes, key = votes.get) # Returns the label with the strongest vote

    def move_file(self, src_path, label):
        if label is None: # Decides destination folder
            dst_dir = review_pile  # Not confident -> review pile
        else:
            dst_dir = os.path.join(restored_dir, label)

        os.makedirs(dst_dir, exist_ok = True) # Create folder if it does not exist

        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        shutil.move(src_path, dst_path) # Moves the file to its new location

    def main(self, datasetB_path):
        for file_name in os.listdir(datasetB_path): # Main loop that processes every file in Dataset B
            if not file_name.lower().endswith((".png", ".jpg", ".jpeg")): # Skip files that are not images
                continue  # Ignore non-image files

            img_path = os.path.join(datasetB_path, file_name)

            vector = self.embed_image(img_path)
            results = self.query_db(vector) # Finds the similar images in ChromaDB
            label = self.decide_label(results) # Decides the final label using similarity logic

            self.move_file(img_path, label) # Moves the file to the appropriate folder

            status = label if label else "Review"
            print(f"[Archived] {file_name} -> {status}")

if __name__ == "__main__":
    chroma_handler = ChromaDBHandler(
        chroma_path = "./chroma_db",
        collection_name = "datasetA_embeddings"
    )

    archivist = Archivist(
        model_path = "./embedding_model.keras",
        chroma_handler = chroma_handler
    )

    archivist.main(datasetB)
