# Assignment 4: Windows Ransomware Early Detection

This is our implementation of the RENTAKA algorithm based on the Zakaria et al. (2024) paper. 

The goal here is to detect ransomware *before* the damage is done. The code filters out the actual encryption-phase APIs and trains 5 machine learning models purely on the "setup phase" API sequences.

## What You Need
* Python 3.10 or newer
* Mac, Linux, or Windows

To get started, open your terminal, navigate to this folder, and install the required libraries:
`pip install -r requirements.txt`

## How to Run the Code

We designed the pipeline to be flexible so you don't have to wait around if you just want to verify that the logic works. You can run it in two ways:

### 1. The Quick Run (Simulated Data)
If you want to test the pipeline and see the outputs immediately, run this:
`python main.py --csv data/VirusShare_simulated.csv`
*This uses a smaller, pre-processed dataset and finishes in about a minute.*

### 2. The Full Evaluation (Raw Dataset)
To run the full 14,600+ sample dataset from KHAS-CCIP:
`python main.py --csv api_sequences_malware_datasets/VirusShare.csv`

**A quick note on the full run:** Expanding 14,000 sequences into individual API columns usually causes memory crashes. To fix this, we wrote a memory-safe dataloader using `CountVectorizer` and sparse matrices to cap the features safely at 2,000. It works perfectly, but the SVM model will still take a few minutes to train on a dataset this size!

## Where to Find the Results
You don't need to save anything manually. Once the script finishes, check the `results/` folder. It automatically generates and saves:
* A `full_results.csv` file with all the accuracy, F1, TPR, and FPR scores.
* High-res PNGs of the ROC curves, confusion matrices, and model comparisons.
* The trained `.pkl` model files (SVM, Random Forest, kNN, Naive Bayes, and J48).
