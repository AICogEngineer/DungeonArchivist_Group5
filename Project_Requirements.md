# Product Requirements Document (PRD): The Dungeon Archivist
(Instructor-Led Chaos)
**Timeline:** Dec 16, 2025 – Jan 5, 2025
**Team:** 2 Developers

## 1. Executive Summary
* **The Scenario:** Your game studio has acquired a massive archive of assets from a defunct developer. The transfer was a disaster: the folder structure was lost, and every file was renamed to a random hash (e.g., `a7f3d9.png`).
* **The Mission:** You have thousands of mixed icons—swords, walls, monsters, and potions—floating in a single "Data Swamp."
* **The Goal:** Build an AI-powered **Auto-Archivist**. It must analyze the visual content of each file and automatically sort them back into clean, labeled folders (`/weapons`, `/environment`, `/enemies`).
* **The Catch:** You will start with a small "Labeled Set" (Dataset A) to teach the AI, and then you must run it against the "Chaos Set" (Dataset B) provided by your Lead Engineer (Instructor) to restore order.

## 2. The Dataset Strategy
### Dataset A: The Training Ground (Clean)
* **Source:** (Dungeon Crawl 32x32 Tiles.)[https://opengameart.org/content/dungeon-crawl-32x32-tiles]
* **Status:** Organized in Dungeon Crawl Stone Soup Full.zip -- 5.7 Mb 
* **Usage:** You will download this yourself. Use it to train your model on the "Ground Truth" of what a "Sword" looks like vs. a "Wall."

### Dataset B: The "Data Swamp" (Chaos)* **Source:** Provided by Instructor.
* **Status:** Disorganized & Scrambled.
* **Content:** A single folder containing thousands of images with filenames like `9c21b5.png`.
* **Usage:** You cannot train on this initially because it has no labels. You must use your model to predict what these items are, move them to folders, and then use that new data to make your model smarter.

### Dataset C: The "Final Boss" (Hidden)
* **Source:** Held by Instructor.
* **Status:** Unknown.
* **Usage:** During your final demo (delivered to you on 1/2), the Instructor will give you a small folder of random icons you have never seen. Your tool must sort them correctly live.

## 3. Technical Architecture

### Phase 1: The Vision Model (Dataset A)
* **Goal:** Teach the eye.
* **Data Pipeline:** Load the organized Dataset A.
* **The Model:** Build a Keras CNN (Functional API).
* **Input:** (32, 32, 3)
* **Embedding Layer:** The model should flatten to a dense vector (e.g., size 64) before the final classification layer.


* **Training:** Train it to classify the base items (e.g., core classes like "Humanoid", "Weapon", "Wall").
* **Vector Database:**
* Generate embeddings for all Dataset A images.
* Store them in ChromaDB with their correct labels as metadata.


### Phase 2: The Restoration (Dataset B)
* **Goal:** Order from chaos.
* **The Task:**
1. Iterate through the files in the Dataset B folder provided by the instructor.
2. Pass each image through your trained Model to get its vector.
3. Query your Vector DB for the **Top 5 Nearest Neighbors**.

* **Auto-Label Logic:**
* **Weighted Voting:** If neighbors are [Sword, Sword, Dagger, Sword, Wall] \rightarrow Label: "Weapon".
* **Confidence Check:** If the neighbors are too dissimilar (high distance), move the file to a `./review_pile/` for human review.

* **Action:** The script automatically moves the file to `./restored_archive/Weapon/9c21b5.png`.

### Phase 3: The "Expansion" Analysis* **Goal:** Improve the brain.
* **The Loop:** Once Dataset B is sorted, you effectively have a larger training set.
* **Task:** Retrain the model on the combined (A + newly sorted B) data.
* **Report:** Generate a visual plot (Loss Curve or Accuracy Matrix) answering: *"Did adding the sorted Chaos data improve the model's confidence?"*

## 4. Deliverables
| File | Description |
| --- | --- |
| **`train.py`** | Trains the Keras model on the known labeled data (Dataset A). |
| **`archivist.py`** | The main engine. Scans a target folder, queries Vector DB, and moves files to sorted subfolders. |
| **`analysis.md`** | A report showing the "Before & After" metrics of your model. |
| **Demo** | Live execution of `archivist.py` on the Instructor's "Final Boss" dataset. |

## 5. Hardware Constraints (CPU Safe)* **Resolution:** Strictly 32x32 pixels.
* **Batch Size:** 32.
* **Embeddings:** Keep vector dimensions small (e.g., 32 or 64 floats).

## 6. Definition of Done* **Code Quality:** The project is broken into a "Training" script and an "App" script (`archivist.py`).
* **Functionality:** The `archivist.py` script successfully moves at least **80%** of the Chaos Dataset into the correct folders (Weapon to Weapon, Wall to Wall).
* **Research:** The `analysis.md` clearly explains how the Vector DB was used to classify the unlabeled images (e.g., *"We used Cosine Similarity with a threshold of 0.2"*).

### **GitHub Workflow**
* **Branching Strategies**: must implement some bratching strategies with frequent pushes, commits and merges to main. Implement branch protections on main...
* **Daily Commit**: Working Days will be Checked for commits please have at least several lines of code worked on. I don't care the frequency or size of commits.
* **Scheduled Code Review**: One of the PRs must be code reviewed with your PM (Trainer) where you will be asked about the choices you've made and what you're code is handling.

### **Capstone Project Presentation (10 Minutes)*** **Problem & Solution (1 min):** Briefly define the user problem and the project's goal.
* **Architecture (2 mins):** High-level overview of the tech stack and data flow.
* **Live Demo (5 mins):** Execute the code live to demonstrate a successful workflow (have a backup recording ready).
* **Q&A (2 mins):** Discuss key technical challenges and answer peer/instructor questions.