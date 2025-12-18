# Dungeon Archivist: AI-Powered Game Asset Sorter

**Timeline:** Dec 16, 2025 – Jan 5, 2025  
**Team:** By Harsh Chavva & Koshik Mahapatra

---

## 1. Executive Summary

### The Scenario
Our game studio inherited a massive archive of visual assets from a defunct developer. During transfer, the original folder structure was destroyed and all filenames were replaced with random hashes (ex: `a7f3d9.png`). Thousands of icons - swords, walls, monsters, potions - now exist in a single chaotic mess of a directory known as the **Data Swamp**.

### The Mission
Build an **AI-powered Dungeon Asset Sorter** capable of analyzing each image and automatically restoring order by organizing the assets into clean, labeled folders such as `/weapons`, `/environment`, or `/enemies`.

### The Goal
Train a vision model on a small, clean dataset and then use it to correctly classify and organize unlabeled assets.

### The Challenge
Only a small labeled dataset is available at the start. The system must:
1. Learn from clean data (Dataset A)
2. Predict labels for chaotic data (Dataset B)
3. Improve itself using newly sorted assets
4. Perform a live test on a hidden evaluation dataset (Dataset C)

---


## 2. Dataset Strategy

### Dataset A - *The Training Ground (Cleaned)*
- **Source:** [Dungeon Crawl 32x32 Tiles](https://opengameart.org/content/dungeon-crawl-32x32-tiles)
  - We used the following zip file: Dungeon Crawl Stone Soup Full

- **Status:** 5.7 MB of organized and labeled data

- **Usage:** Train the initial convolutional neural network (CNN) and create trusted reference embeddings

- **Notes:** Downloaded manually by the team


### Dataset B - *The Data Swamp (Chaos)*
- **Source:** Provided by our Supervisor

- **Status:** Files are mixed together, unlabeled, and named with random hashes

- **Usage:**
  - Run through the trained model
  - Predict image categories by comparing visual similarity
  - Auto-sort into the correct folders
  - Newly labeled images are used for model expansion


### Dataset C - *The Final Boss (Hidden)*
- **Source:** Supervisor (delivered on 1/2)

- **Status:** Unknown

- **Usage:** Use this dataset in a live demo to check how well the model handles new, unseen images

---


## 3. Technical Architecture

### Phase 1 - Vision Model (Dataset A)
**Goal:** Teach the model how to recognize visual patterns.

- **Data Pipeline:** Load organized Dataset A

- **Model:** A CNN built using Keras’ Functional API

- **Input Shape:** `(32, 32, 3)`

- **Embedding Layer:** Flatten the image features and pass them through a dense layer to get a compact embedding of 32–64 floats per image

- **Training Objective:** Classify data into the base categories  
  - Example: `Weapon`, `Wall`, `Humanoid`


#### Vector Database
- Generate embeddings for all Dataset A images
- Save these embeddings in **ChromaDB**
- Store the correct label as metadata with each embedding

---


### Phase 2 - Restoration (Dataset B)

**Goal:** Restore order from chaos by sorting all unordered assets.

**Process:**
1. Iterate through all images in Dataset B
2. Generate embeddings using the trained CNN
3. Query ChromaDB for **Top-5 nearest neighbors**
4. Apply **auto-label logic**

**Auto-Label Logic:**
- **Weighted Voting:**
  - Example: `[Sword, Sword, Dagger, Sword, Wall] -> Weapon`

- **Confidence Check:**
  - If similarity confidence is low, move the image to `./review_pile/`

- **Action:**
  - Confident predictions are moved to  
    `./restored_archive/<Label>/filename.png`

---


### Phase 3 - Expansion Analysis
**Goal:** Improve model accuracy using the expanded dataset

- Combine Dataset A with newly sorted Dataset B
- Retrain the model using the expanded dataset
- Compare results before and after retraining
- Show performance changes using visual charts (loss curves or accuracy plots)

---


## 4. Deliverables

| File | Description |
|-----|------------|
| `train.py` | Trains the CNN on Dataset A and any newly sorted data to improve accuracy |
| `archivist.py` | Main program that creates embeddings, looks up similar images in the vector database, and automatically sorts files into folders |
| `analysis.md` | Report showing model performance before and after expansion, including confidence, accuracy, and threshold logic |
| Demo | Live run of `archivist.py` on the hidden evaluation dataset provided by the supervisor |

---


## 5. Hardware Constraints (CPU-Friendly)

- **Resolution:** Only 32 × 32 pixels

- **Batch Size:** Process 32 images at a time

- **Embedding Size:** Each image is represented by 32 - 64 floats

- **Hardware:** Can run entirely on a CPU, no GPU required

---


## 6. Project Completion Outcome

By the end of the project, the Dungeon Asset Sorter will:
- Maintain clear separation between training (`train.py`) and application logic (`archivist.py`)
- Automatically sort at least **80%** of Dataset B into the correct category folders
- Use `analysis.md` to describe how the vector database is used, how similarity between images is measured, and how the confidence cutoff (threshold) determines which images go to the `./review_pile/`
- Visually understand game assets
- Restore order to chaotic data automatically
- Improve through self-expansion
- Perform reliably on unseen datasets in a live demo

**From a Data Swamp to a fully Structured Archive, now completely automated.**