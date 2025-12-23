# Trains the Keras model on the known labeled data

import numpy as np
import tensorflow as tf
from tensorflow import keras

import chromadb

def __init__(self, artifacts_dir, chroma_path, collection_name):
    self.artifacts_dir = artifacts_dir
    self.chroma_path = chroma_path
    self.collection_name = collection_name

    self.client = chromadb.PersistentClient(path=self.chroma_path)
    self.collection = self.client.get_or_create_collection(
    name=self.collection_name,
        metadata={"hnsw:space": "cosine"},
    )





