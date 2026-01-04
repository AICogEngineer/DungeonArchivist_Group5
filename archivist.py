# The main engine. Scans a target folder, queries Vector DB, and moves files to sorted subfolders.

"""
ARCHIVIST.PY — PHASE 2: THE RESTORATION ENGINE

Purpose:
- Takes the unlabeled images (from Dataset B & Dataset C)
- Convert each image into an embedding using the trained CNN
- Query ChromaDB for nearest neighbors
- Decide a label using similarity + voting
- Move files into restored folders or a review pile
"""

import os
# Suppresses the TensorFlow INFO logs and oneDNN messages (doesn't affect the code's correctness)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # Suppresses unnecessary TensorFlow logs
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # Prevents CPU-specific warnings

import shutil # to move files across folders
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import models
import chromadb
import datetime # Used to timestamp TensorBoard log directories
from collections import defaultdict # For weighted voting accumulation

# Dataset B (Random dataset)
    # Getting dataset do below or if that doesn't work: Open File Explorer, open the dataset, right click the dataset name, select Copy address as text, and paste that below
datasetB = os.environ.get(
    "DATASET_B_DIR",
    "./ChaosData_DatasetB"
)

# Dataset C (Unknown dataset) is a local clone of a GitHub repo
    # Before each run: git clone https://github.com/AICogEngineer/dataset_c.git
datasetC = os.environ.get(
    "DATASET_C_DIR",
    "./dataset_c" # The local repo path after git cloning it
)

# Output folders
restored_dir = "./Restored_Archive" # Where classified images will be moved, if model is confident
review_pile = "./Review_Pile" # Where images go if the model is uncertain about the results

IMG_SIZE = (32, 32) # Must match the training image size in train.py
k = 5 # Number of nearest neighbors to check
threshold = 0.35 # Max average cosine distance and rejects globally weak matches (0 = identical, higher = less similar)
                    # ChromaDB returns cosine distance because embeddings are L2-normalized
label_str_ratio = 0.6 # Represents >= 60% of vote strength and prevents split-vote ambiguity
margin_threshold = 0.15 # The margin is to prevent near-tie false confidence

# Sets up TensorBoard
log_dir = os.path.join( # Creates a unique log directory per run so graphs don’t overwrite
    "logs", # Root logging directory
    "archivist", # Project-specific subfolder
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S") # To make a unique log directory with timestamp
) # After running the file, in terminal run: tensorboard --logdir=logs/archivist

summary_writer = tf.summary.create_file_writer(log_dir) # Creates a TensorBoard writer that will log per-image metrics

def load_image(path):
    image_bytes = tf.io.read_file(path)
    image = tf.image.decode_image( # Force RGBA so alpha handling is consistent even if source image lacks transparency
        image_bytes,
        channels = 4,
        expand_animations = False
    )
    image = tf.image.resize(image, IMG_SIZE) # Resizes the image to 32 by 32 so it matches the model input
    # In train.py I composited alpha onto white so I have to do that here too
    rgb = tf.cast(image[..., :3], tf.float32) / 255.0
    alpha = tf.cast(image[..., 3:4], tf.float32) / 255.0
    bg = tf.ones_like(rgb)
    image = rgb * alpha + bg * (1.0 - alpha)

    return image

class Archivist:
    def __init__(self):
        self.model = models.load_model("./embedding_model.keras") # Loads the embedding model
        self.client = chromadb.PersistentClient("./chroma_db") # Connect to the persistent vector DB (ChromaDB) instance
        self.collection = self.client.get_or_create_collection(
            "datasetA_embeddings"
        )

        # Initialize counters for final results
        self.total_images = 0
        self.restored_images = 0
        self.review_images = 0
        self.distances = [] # To track similarity quality across all images

        # Makes sure the output folders actually exist
        os.makedirs(restored_dir, exist_ok = True)
        os.makedirs(review_pile, exist_ok = True)

    def embed_image(self, image_path):
        img = tf.expand_dims(load_image(image_path), axis = 0) # Adds a batch dimension because Keras wants it

        vector = self.model.predict(img, verbose = 0)[0].astype(np.float32) # Generates the embedding vector from the CNN

        # Normalizes embedding for correct cosine similarity
        norm = np.linalg.norm(vector) # This normalization is different from train.py because this is safer since here we may have to worry that images could be fully transparent, single-color or corrupted
        if norm > 0:
            vector /= norm  # Without this, distances would be inconsistent

        return vector
    
    # Handles Confidently Classified Files
    def restore_file(self, src_path, file, label):
        dst_dir = os.path.join(restored_dir, label) # Makes the destination directory path
        os.makedirs(dst_dir, exist_ok = True) # Creates the label folder if it doesn’t exist (prevents errors on first use)

        dst_path = os.path.join(dst_dir, file) # Creates the destination file path for the image

        if os.path.exists(dst_path): 
            file = f"{datetime.datetime.now().timestamp()}_{file}" # Adds a timestamp prefix to ensure the filename is unique and prevents overwriting
            dst_path = os.path.join(dst_dir, file) # Remakes the destination path using the new unique filename

        shutil.move(src_path, dst_path)
        self.restored_images += 1
    
    # Handles Uncertain / Low-Confidence Files
    def send_to_review(self, src_path, file):
        os.makedirs(review_pile, exist_ok = True) # Makes sure the review directory exists before moving files into it
        dst_path = os.path.join(review_pile, file)
 
        if os.path.exists(dst_path): # Prevents overwriting if the same file name already exists in the review pile
            file = f"{datetime.datetime.now().timestamp()}_{file}" # Appends a timestamp to create a unique filename
            dst_path = os.path.join(review_pile, file)

        shutil.move(src_path, dst_path)
        self.review_images += 1

    # Processes a dataset folder by finding images, embedding them, querying them in ChromaDB, and sorting them into the restored or review folders so that way it works with both dataset B & C
    def process_dataset(self, dataset_path, dataset_name): # Walks dataset recursively, embeds each image, queries Dataset A and decides whether to go with the restore or manual review folder
        print(f"\nProcessing {dataset_name} . . . ") # For user reference

        for root, _, files in os.walk(dataset_path): # Recursive scan of the dataset's path to find all images
            root = os.path.abspath(root)
            if root.startswith(os.path.abspath(restored_dir)) or root.startswith(os.path.abspath(review_pile)): # Normalize paths to avoid reprocessing on Windows due to path separator differences
                continue
            
            for file in files:
                if not file.lower().endswith((".png", ".jpg", ".jpeg")): # Skips non-image files
                    continue

                self.total_images += 1 # Counts total processed images
                src_path = os.path.join(root, file) # Builds the full file path to the image

                vector = self.embed_image(src_path) # Converts the image into an embedding vector

                # Query ChromaDB to find the k most similar embeddings from Dataset A
                results = self.collection.query(
                    query_embeddings = [vector.tolist()], # Converts the NumPy embedding into a plain list and sends it as the query vector to ChromaDB
                    n_results = k, # Asks ChromaDB to return the k nearest neighbors
                    include = ["metadatas", "distances"]
                )

                distances = results["distances"][0] # Gets the list of distances for all k neighbors
                labels = [meta["label"] for meta in results["metadatas"][0]] # Pulls the "label" field from each metadata object

                avg_distance = float(np.mean(distances)) # Finds the average distance across all k neighbors
                self.distances.append(avg_distance) 

                # Confidence Check
                if avg_distance > threshold: # If the distance is too high (average similarity is too weak), the model is not confident enough to auto-label the image
                    self.send_to_review(src_path, file)
                    continue

                # Weighted K-NN Voting
                vote_scores = defaultdict(float) # Accumulates weighted votes per label

                for rank, (label, dist) in enumerate(zip(labels, distances)): # zip() pairs each label with its distance
                    weight = (1 / (dist + 1e-6)) * (1 / (rank + 1)) # Combines semantic similarity (distance) and neighbor reliability (rank)
                    # Closer neighbors contribute to more of the vote weight
                    vote_scores[label] += weight

                sorted_votes = sorted(vote_scores.items(), key = lambda x: x[1], reverse = True)

                if len(sorted_votes) < 2:
                    self.send_to_review(src_path, file)
                    continue

                (best_label, best), (_, second) = sorted_votes[:2]

                # 2nd confidence check to prevent ambiguous labeling
                if best - second < margin_threshold: # edge case
                    self.send_to_review(src_path, file)
                    continue

                if best / sum(vote_scores.values()) < label_str_ratio:
                    self.send_to_review(src_path, file)
                    continue

                self.restore_file(src_path, file, best_label) # If all confidence checks pass, restore file

                # TensorBoard logging allows post-run analysis of model confidence trends
                with summary_writer.as_default(): # Logs metrics so TensorBoard can visualize model behavior over time
                    tf.summary.scalar( # Logs the average k-NN distance for this image
                        f"{dataset_name} / nn_distance", # To make it easier to read since there are multiple datasets
                        avg_distance,
                        step = self.total_images # X-axis value in TensorBoard
                    )
                    tf.summary.scalar( # Logs how many images have been processed so far
                        f"{dataset_name} / processed_images",
                        self.total_images,
                        step = self.total_images
                    )

    def process(self): # Entry point that processes all unlabeled datasets sequentially
        self.process_dataset(datasetB, "dataset_B")
        self.process_dataset(datasetC, "dataset_C")

        # Logs the final metrics once processing is complete
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
        print(f"\nConfidently classified coverage: {(self.restored_images / self.total_images) * 100:.2f}%") # Percentage of images confidently classified
        print(f"Flagged for manual review rate: {(self.review_images / self.total_images) * 100:.2f}%") # Percentage of images flagged for manual review
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
# After running the file, in terminal run: tensorboard --logdir=logs/archivist