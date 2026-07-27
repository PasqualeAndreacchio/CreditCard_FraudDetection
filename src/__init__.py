"""
Credit Card Fraud Detection Project.
This project aims to detect fraudulent credit card transactions using deep learning techniques.

Modules:
    - Datasets/preprocess.py: Preprocessing pipeline (cleaning, splitting, scaling, SMOTE)
    - Datasets/datasets.py: ContrastiveDataset for triplet-based learning
    - models/Autoencoder.py: Encoder, Decoder, FraudAutoencoder, ContrastiveModel
    - models/Classifier.py: FraudDetectionMLP (supervised binary classifier)
    - Train/trainer.py: Generic training pipeline with early stopping and checkpointing
    - Train/contrastive_trainer.py: Contrastive pre-training with TripletMarginLoss
    - Evaluation/: Evaluation, metrics, and plotting modules
    - utils.py: Configuration loading, seeding, device, and logging
"""

__authors__ = ["Andreacchio Pasquale", 
               "Calcagni Matteo Renato",
               "Lavarda Nicola"]
__email__ = ["pasquale.andreacchio@studenti.unipd.it",
             "matteorenato.calcagni@studenti.unipd.it",
             "nicola.lavarda.1@studenti.unipd.it"]
