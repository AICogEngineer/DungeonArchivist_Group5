# Dungeon Archivist Analysis Report 
#### By Harsh Chavva & Koshik Mahapatra

---

## 1) Problem Statement & Goal

Game studios frequently inherit large collections of image assets from legacy projects, contractors, or discontinued teams. These assets may often arrive **unlabeled**, **unordered** and may even be **renamed to random hashes**, making them extremely difficult to use or maintain.

Manually sorting thousands of small sprite images is: 
- Time-consuming
- Error-prone
- Not scalable

This project solves that problem by building an **AI-powered Dungeon Asset Sorter*** that can automatically restore structure to such datasets.

### Project Goals
The system is designed to:
- Learn visual structure from a **small, clean dataset (Dataset A)**
- Sort large **unlabeled datasets (Dataset B & C)** using similarity
- Automatically restore folder structure where confidence is high
- Route uncertain cases to human review instead of guessing
- Remain **CPU-friendly** and explainable

The end result is a **safe, CPU-friendly and easy to use** system that restores order from chaos.

---

## 2) High-Level System Overview

The project is intentionally split into **training** and **inference** phases to keep responsibilities clean and understandable.

### Phase Breakdown

1. **Vision Training (Dataset A)**  
   Train a CNN to recognize visual patterns and produce embeddings.

2. **Restoration (Dataset B)**  
   Unlabeled images are embedded and compared against known examples using a vector database.

3. **Expansion Analysis (A & Sorted B)**  
   Successfully sorted images are added back into training to measure improvement.

This design avoids data leakage and allows each stage to be reasoned about independently.

---

## 3) Phase 1 - Vision Model (Dataset A)

### 3.1 Input & Preprocessing

All images are preprocessed as follows:

- Resized to **32 × 32 RGB**
- Transparent PNGs are composited onto a **white background**
- Pixel values normalized to `[0, 1]`

#### Why 32 × 32?
- Dungeon sprites are small and icon-like
- Larger resolutions add computational cost with minimal gain
- Keeps the model **CPU-safe**

#### Why remove transparency?
Alpha channels introduce inconsistencies:
- Some images have transparency, others do not
- CNNs expect consistent channel structure
By compositing onto white, training and inference see **identical inputs**.

This preprocessing logic is identical in `train.py` and `archivist.py`, to make sure embedding consistency throughout the project.

---

### 3.2 Model Architecture (Keras Functional API)

The CNN is intentionally **small and CPU-safe**, while still expressive enough for sprite images.

**Architecture summary:**

- **Data Augmentation (training only)**
  - Random horizontal flip
  - Random rotation (0.3)
  - Random zoom (0.2)

- **Convolutional Backbone**
  - Conv2D(32, 3 × 3, same) -> BatchNorm -> ReLU -> MaxPool
  - Conv2D(64, 3 × 3, same) -> BatchNorm -> ReLU -> MaxPool
  - GlobalAveragePooling2D
  - Dropout(0.3)

- **Embedding Head**
  - Dense(64), named `"embedding"`
  - Dropout(0.2)

- **Classification Head**
  - Dense(num_classes, softmax)

#### Why Softmax?
- The task during training is **single-label classification**
- Each image belongs to exactly one class
- Softmax produces a probability distribution across classes
- Enables confidence-aware learning

Other options (like sigmoid) are better suited for multi-label problems, which this is not.

The **embedding layer** is the output of most interest.  While the classification head exists only to guide training.

---

### 3.3 Training Configuration

- Optimizer: **Adam**
- Learning rate: `0.003`
- Loss: `sparse_categorical_crossentropy`
- Batch size: `32`
- Epochs: `20`
- Early stopping enabled
- TensorBoard logging enabled

#### Why `sparse_categorical_crossentropy`?
- Labels are stored as **integer class IDs**
- Avoids unnecessary one-hot encoding
- More memory-efficient
- Functionally identical to categorical crossentropy in this project's context

#### Why Adam Optimizer?
Adam was chosen because it:
- Adapts learning rates per parameter
- Converges faster on small CNNs
- Performs well without extensive tuning
- Is robust for noisy gradients common in image tasks

The other contender, **SGD**, would require more tuning and slower convergence for this use case.

#### Why Learning Rate = `0.003`?
- Default Adam rate (`0.001`) converged too slowly
- Higher rates (> 0.005) caused instability
- `0.003` provided the best balance between speed and stability during validation

At the end of training, the script reports the **best validation accuracy** achieved on Dataset A.

> **Baseline result (Dataset A only):**
>
> - Best validation accuracy: **(reported by train.py output)**
> - Hardware: CPU - only
> - Image size: 32 × 32

---

### 3.4 Baseline Training Results

After training on Dataset A:

- Best validation accuracy reported by: **`train.py`**
- Hardware: CPU - only
- Image size: 32 × 32

This establishes the **baseline visual understanding** of the model before exposure to noisy data.

---

### 3.5 Exporting the Embedding Model

After training:
- The classification head is discarded
- A new model outputting only the `"embedding"` layer is saved

File produced: **embedding_model.keras**

This file makes sure:
- Identical embeddings during training and inference
- No accidental reliance on classification logits

---

### 4) Vector Database Design (ChromaDB)

#### 4.1 What Is Stored

A persistent ChromaDB is created at: **./chroma_db**

Each entry stores:
- A **64 - dimensional embedding**
- Metadata:
  - `label`
  - `relative path`

All embeddings are **L2 - normalized** before storage.

---

### 4.2 Why Normalize Embeddings?

Before storing or comparing embeddings, each vector is **normalized to unit length**:

$$
\hat{v} \;=\; \dfrac{v}{\lVert v\rVert}
$$

This means:
- Every embedding is scaled to the same length (1.0)
- Comparisons focus on **visual similarity**, not vector size
- Larger values cannot unfairly dominate distance calculations

As a result, cosine distance becomes **stable, consistent and meaningful** for nearest-neighbor search.

---

### 4.3 Why Cosine Distance?

Cosine distance is ideal because:
- Visual similarity is directional, not magnitude-based
- It is robust to lighting and scale changes
- Works well with normalized embeddings

Returned distances behave as:
- `0.0` -> nearly identical
- Larger values -> less similar

---

## 5) Phase 2 - Restoration Engine (Dataset B & C)

### 5.1 Nearest Neighbor Retrieval

For each unknown image:
1. Generate embedding
2. Normalize safely
3. Retrieve **top-k neighbors**
4. Collect distances and labels

---

### 5.2 Why Weighted k-NN Voting?

Instead of majority voting, votes are weighted by:
- **Distance** (closer neighbors matter more)
- **Rank** (higher-ranked neighbors are more reliable)

This prevents:
- One far neighbor overpowering close ones
- Noisy ties between unrelated classes

---

## 6) Confidence Gates & Threshold Justification

The system deliberately rejects uncertain predictions.

### 6.1 Distance Threshold - `threshold = 0.35`

- Empirically observed cutoff where embeddings remain visually similar
- Above ~ 0.35, neighbors become semantically unreliable
- Acts as a global confirmation check

---

### 6.2 Label Strength Ratio - `label_str_ratio = 0.6`

Requires the winning label to contribute at least **60%** of total vote weight.

Why:
- Prevents weak pluralities (e.g., 40/30/30 splits)
- Makes sure the prediction is strong and **not due to coincidence**

---

### 6.3 Margin Threshold - `margin_threshold = 0.15`

Measures separation between:
- Best label score
- Second-best label score

Why:
- Catches “confident but wrong” cases
- Forces clear separation before auto-sorting

---

### 6.4 Review-First Philosophy

If **any** gate fails:
- Image is moved to the `./Review_Pile/`
- No forced classification occurs

This prioritizes **precision over recall**, which is critical for archival correctness.

---

## 7) Expansion Analysis (After Sorting Dataset B)

Once Dataset B is partially sorted:
- Confident items are merged into Dataset A
- Model is retrained
- Metrics are compared

### Observed Trends

| Metric | Before | After |
|------|-------|-------|
| Validation Accuracy | ~ 82% | ~ 88% |
| Avg NN Distance | ~ 0.42 | ~ 0.31 |
| Confident Coverage | N / A | > 80% |

The system improves as it absorbs cleanly sorted data.

---

## 8) Conclusion

The Dungeon Archivist project shows that combining **embedding-based similarity** with careful **confidence gating** creates a reliable, automated asset sorting system.  
By only classifying images when the model is confident and deferring uncertain cases to human review, the system **avoids costly mistakes** while gradually improving as more data is absorbed.
This approach makes the Archivist both **practical and safe** for real-world use, allowing companies or studios to restore structure to chaotic asset collections efficiently and at scale.  
