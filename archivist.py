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
# Suppresses the TensorFlow INFO logs and oneDNN messages (doesn't affect the code's correctness)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import shutil # to move files across folders
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models
import chromadb
import datetime # Used to timestamp each run so TensorBoard logs from different executions do not overwrite each other

# Getting dataset do below or if that doesn't work: Open File Explorer, open the dataset, right click the dataset name, select Copy address as text, and paste that below
datasetB = os.environ.get(
    "DATASET_B_DIR",
    "./ChaosData_DatasetB"
)

# Output folders
restored_dir = "./Restored_Archive" # Where classified images will be moved, if model is confident
review_pile = "./Review_Pile" # Where images go if the model is uncertain about the results

IMG_SIZE = (32, 32) # Must match the training image size in train.py
k = 5 # Number of nearest neighbors to check
threshold = 0.35 # Confidence cutoff (Cosine distance: lower = more similar, higher = less confident)

# Sets up TensorBoard
log_dir = os.path.join(
    "logs", # Root logging directory
    "archivist", # Project-specific subfolder
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # Unique ID
) # In terminal run: tensorboard --logdir=logs/archivist

summary_writer = tf.summary.create_file_writer(log_dir) # Creates a TensorBoard writer that will store performance metrics

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

class Archivist:
    def __init__(self):
        self.model = models.load_model("./embedding_model.keras") # Loads the embedding model
        self.client = chromadb.PersistentClient("./chroma_db") # Connect to the vector DB
        self.collection = self.client.get_or_create_collection(
            "datasetA_embeddings"
        )

        # Initialize counters for final results
        self.total_images = 0
        self.restored_images = 0
        self.review_images = 0
        self.distances = []

        # Makes sure the output folders actually exist
        os.makedirs(restored_dir, exist_ok = True)
        os.makedirs(review_pile, exist_ok = True)

    def embed_image(self, image_path):
        img = tf.expand_dims(load_image(image_path), axis = 0) # Adds a batch dimension because Keras wants it

        return self.model.predict(img, verbose = 0)[0].astype(np.float32) # Generates the embedding vector

    def process(self):
        for root, _, files in os.walk(datasetB): # Recursive scan of Dataset B to find all images
            for file in files:
                if not file.lower().endswith((".png", ".jpg", ".jpeg")): # Skips non-image files
                    continue

                self.total_images += 1 # Counts total processed images
                src_path = os.path.join(root, file) # Builds the full file path to the image

                vector = self.embed_image(src_path) # Converts the image into an embedding vector

                results = self.collection.query( # Asks ChromaDB to find the k closest embeddings
                    query_embeddings = [vector.tolist()],
                    n_results = k,
                    include = ["metadatas", "distances"]
                )

                best_distance = results["distances"][0][0] # Finds the distance to the closest known image
                self.distances.append(best_distance) # Saves the distance for statistics summary at the end

                if best_distance > threshold: # If similarity is too low, sends images to the review pile
                    shutil.move(src_path, os.path.join(review_pile, file))
                    self.review_images += 1
                else: # Restores images to predicted category folder
                    rel_label = results["metadatas"][0][0]["label"]

                    dst_dir = os.path.join(restored_dir, rel_label) # Creates the destination folders if they don’t already exist
                    os.makedirs(dst_dir, exist_ok = True)

                    shutil.move(src_path, os.path.join(dst_dir, file)) # Moves images to its restored location
                    self.restored_images += 1

        self.log_tensorboard_metrics()
        self.print_results()

    # Logs final performance metrics to TensorBoard
    def log_tensorboard_metrics(self):
        coverage_rate = self.restored_images / self.total_images # Percentage of images confidently classified
        review_rate = self.review_images / self.total_images # Percentage of images sent for manual review
        avg_distance = float(np.mean(self.distances)) # Average cosine distance to nearest known embedding (Lower values are better since it means a higher similarity)

        with summary_writer.as_default():
            tf.summary.scalar("coverage_rate", coverage_rate, step = 0) # How much of Dataset B was successfully restored
            tf.summary.scalar("review_rate", review_rate, step = 0) # How often the model was uncertain
            tf.summary.scalar("avg_nn_distance", avg_distance, step = 0) # Overall embedding similarity quality
            tf.summary.scalar("processed_images", self.total_images, step = 0) # Total number of images processed in this run

            summary_writer.flush() # Ensures data is visible when you open TensorBoard

    def print_results(self):
        print("\n   ARCHIVIST RESULTS SUMMARY")
        print(f"In total processed: {self.total_images} images")
        print(f"Sent {self.restored_images} images to the Restored Archive")
        print(f"Sent {self.review_images} images to the Review Pile")
        print(f"\nClassification coverage: {(self.restored_images / self.total_images) * 100:.2f}%") # Percentage of images confidently classified
        print(f"Review rate: {(self.review_images / self.total_images) * 100:.2f}%") # Percentage of images flagged for manual review
        print(
            f"Avg NN distance: {np.mean(self.distances):.4f}"
            " (average similarity distance to nearest known image)"
        )

# Runs the archivist when the script is executed directly
try:
    if __name__ == "__main__":
        Archivist().process()
except ZeroDivisionError:
    print("\nThere's NOTHING to sort, your Dataset is empty.")