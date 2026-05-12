import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# ==========================================
# CONFIGURAZIONE (Iperparametri LSTM)
# ==========================================
CARTELLA_INPUT = "output"
CARTELLA_MODELLI = "modelli"


FINESTRA = 30
STRIDE = 15
NUM_FEATURES = 34
HIDDEN_1 = 128
HIDDEN_2 = 64
DENSE_1 = 32
DENSE_2 = 16
DROPOUT = 0.3
NUM_CLASSI = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHE = 50
PAZIENZA = 10


def crea_sequenze_da_csv(percorso_csv):
    df = pd.read_csv(percorso_csv)
    
    sequenze_video = []

    for id_persona, dati_persona in df.groupby('id_persona'):
        coordinate = dati_persona.iloc[:, 2:].values

        for i in range(0, len(coordinate) - FINESTRA + 1, STRIDE):
            fetta_video = coordinate[i : i + FINESTRA]
            sequenze_video.append(fetta_video)

    return sequenze_video


def main():

    sequenze = crea_sequenze_da_csv("output/train/fight/fight_001.csv")
    
    print(f"Ho estratto {len(sequenze)} sequenze in totale.")
    if len(sequenze) > 0:
        print(f"Forma di una singola sequenza: {sequenze[0].shape}") 
        # Dovrebbe stampare: (30, 34)

    

if __name__ == "__main__":
    main()