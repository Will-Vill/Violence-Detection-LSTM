import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if torch.backends.mps.is_available():
    torch.mps.manual_seed(SEED)

dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Viene utilizzato il dispositivo: {dispositivo}")

CARTELLA_INPUT = "tipi_output_yolo/output_YOLOnNANO"
CARTELLA_MODELLI = "modelli"

FINESTRA = 30
STRIDE = 15

# FEATURES (per frame):
#   Coordinate relative normalizzate:  34  (pose pura, no posizione)
#   Velocità delle coord. relative:    34
#   Distanza dal vicino:               1
#   TOTALE:                            69
NUM_FEATURES = 69

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

# Indici keypoints COCO (usati da YOLO-Pose)
# 0:naso, 1:occhio_sx, 2:occhio_dx, 3:orecchio_sx, 4:orecchio_dx,
# 5:spalla_sx, 6:spalla_dx, 7:gomito_sx, 8:gomito_dx,
# 9:polso_sx, 10:polso_dx, 11:anca_sx, 12:anca_dx,
# 13:ginocchio_sx, 14:ginocchio_dx, 15:caviglia_sx, 16:caviglia_dx
#
# Nel vettore a 34: indice_x = kp_id*2, indice_y = kp_id*2+1


def normalizza_keypoints(coords):
    """
    Trasforma le coordinate assolute in coordinate relative normalizzate.

    1. Calcola il centro del corpo (media di tutte le coordinate)
    2. Sottrae il centro → la posa diventa indipendente dalla posizione
    3. Divide per l'altezza del corpo (naso→centro_anche)
       → la posa diventa indipendente dalla scala (lontano/vicino dalla cam)

    Risultato: un vettore che descrive SOLO la posa del corpo,
    indipendentemente da dove si trova nel frame o quanto è grande.
    """
    x_coords = coords[0::2].copy()
    y_coords = coords[1::2].copy()

    # Centro del corpo
    cx = np.mean(x_coords)
    cy = np.mean(y_coords)

    # Altezza del corpo: distanza naso → centro delle anche
    naso_x, naso_y = coords[0], coords[1]
    anca_cx = (coords[22] + coords[24]) / 2  # media anca_sx e anca_dx (x)
    anca_cy = (coords[23] + coords[25]) / 2  # media anca_sx e anca_dx (y)
    altezza = np.sqrt((naso_x - anca_cx)**2 + (naso_y - anca_cy)**2)

    # Evita divisione per zero (keypoints non rilevati)
    if altezza < 0.01:
        altezza = 0.2  # valore di default ragionevole

    # Coordinate relative normalizzate
    x_rel = (x_coords - cx) / altezza
    y_rel = (y_coords - cy) / altezza

    # Ricomponi il vettore interleaved [x0,y0, x1,y1, ...]
    risultato = np.empty(34)
    risultato[0::2] = x_rel
    risultato[1::2] = y_rel

    return risultato


def crea_sequenze_da_csv(percorso_csv):
    """
    Come il modello base, ma con coordinate RELATIVE NORMALIZZATE.

    La differenza fondamentale: nel base le coordinate erano assolute
    (posizione nel frame 0-1), quindi la stessa posa in punti diversi
    del frame appariva diversa alla rete. Ora le coordinate sono
    relative al centro del corpo e normalizzate per l'altezza,
    rendendo la rete invariante a posizione e scala.
    """
    df = pd.read_csv(percorso_csv)

    if df.empty:
        return []

    sequenze_video = []

    # Pre-calcolo centri per la distanza
    centri_per_frame = {}
    for frame_num, dati_frame in df.groupby('frame'):
        centri = {}
        for _, riga in dati_frame.iterrows():
            pid = int(riga['id_persona'])
            coords = riga.iloc[2:].values.astype(float)
            x_c = coords[0::2]
            y_c = coords[1::2]
            centri[pid] = np.array([np.mean(x_c), np.mean(y_c)])
        centri_per_frame[frame_num] = centri

    for id_persona, dati_persona in df.groupby('id_persona'):
        coordinate_raw = dati_persona.iloc[:, 2:].values
        lista_frame = dati_persona['frame'].values

        if len(coordinate_raw) < 2:
            continue

        # Normalizza le coordinate per ogni frame
        coordinate_norm = np.array([
            normalizza_keypoints(row) for row in coordinate_raw
        ])

        # Velocità sulle coordinate normalizzate
        velocita = np.diff(coordinate_norm, axis=0)
        velocita = np.vstack([np.zeros((1, 34)), velocita])

        # Distanza dal vicino più prossimo (come nel base)
        distanze = []
        for frame_num in lista_frame:
            centri = centri_per_frame.get(frame_num, {})
            if len(centri) < 2 or id_persona not in centri:
                distanze.append(1.0)
            else:
                centro_persona = centri[id_persona]
                min_dist = float('inf')
                for altro_id, altro_centro in centri.items():
                    if altro_id != id_persona:
                        dist = np.sqrt(np.sum((centro_persona - altro_centro)**2))
                        if dist < min_dist:
                            min_dist = dist
                distanze.append(min_dist)

        distanze = np.array(distanze).reshape(-1, 1)

        # coord_norm(34) + vel(34) + dist(1) = 69
        features = np.hstack([coordinate_norm, velocita, distanze])

        # Sliding window per persona
        for i in range(0, len(features) - FINESTRA + 1, STRIDE):
            sequenze_video.append(features[i : i + FINESTRA])

    return sequenze_video


def carica_dataset(split):
    X, y = [], []
    for categoria in ["fight", "no_fight"]:
        etichetta = 1 if categoria == "fight" else 0
        cartella = os.path.join(CARTELLA_INPUT, split, categoria)
        for file_csv in glob.glob(os.path.join(cartella, "*.csv")):
            sequenze = crea_sequenze_da_csv(file_csv)
            print(f"  {categoria}: {len(sequenze)} sequenze da {file_csv}")
            for seq in sequenze:
                X.append(seq)
                y.append(etichetta)
    return np.array(X), np.array(y)


class FightDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMClassificatore(nn.Module):
    """
    Stessa architettura del modello base.
    2 LSTM + 3 Fully Connected con Dropout.
    """
    def __init__(self):
        super(LSTMClassificatore, self).__init__()
        self.lstm1 = nn.LSTM(input_size=NUM_FEATURES, hidden_size=HIDDEN_1, batch_first=True)
        self.dropout1 = nn.Dropout(DROPOUT)
        self.lstm2 = nn.LSTM(input_size=HIDDEN_1, hidden_size=HIDDEN_2, batch_first=True)
        self.dropout2 = nn.Dropout(DROPOUT)
        self.fc1 = nn.Linear(HIDDEN_2, DENSE_1)
        self.dropout3 = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(DENSE_1, DENSE_2)
        self.dropout4 = nn.Dropout(DROPOUT)
        self.fc3 = nn.Linear(DENSE_2, NUM_CLASSI)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = x[:, -1, :]
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.dropout4(x)
        x = self.fc3(x)
        return x


def main():
    print("Caricamento Train Set...")
    X_train, y_train = carica_dataset("train")
    print("\nCaricamento Validation Set...")
    X_val, y_val = carica_dataset("val")

    if len(X_train) == 0 or len(X_val) == 0:
        print("ERRORE: Non ci sono abbastanza dati")
        return

    print(f"\nX_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}    y_val:   {y_val.shape}")
    print(f"Distribuzione train → fight: {sum(y_train==1)}, no_fight: {sum(y_train==0)}")
    print(f"Distribuzione val   → fight: {sum(y_val==1)}, no_fight: {sum(y_val==0)}")

    # Niente StandardScaler — le coordinate sono già normalizzate
    # dal preprocessing (relative + divise per altezza corpo)

    dataset_train = FightDataset(X_train, y_train)
    dataset_val = FightDataset(X_val, y_val)
    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)

    modello = LSTMClassificatore().to(dispositivo)
    criterio_loss = nn.CrossEntropyLoss()
    ottimizzatore = optim.Adam(modello.parameters(), lr=LEARNING_RATE)

    storico_loss_train = []
    storico_loss_val = []
    storico_accuracy_val = []
    miglior_val_loss = float('inf')
    contatore_pazienza = 0

    os.makedirs(CARTELLA_MODELLI, exist_ok=True)
    percorso_modello = os.path.join(CARTELLA_MODELLI, "lstm_risse_relativo.pt")

    print(f"\n{'='*60}")
    print(f" Addestramento con COORDINATE RELATIVE — max {EPOCHE} epoche")
    print(f"{'='*60}\n")

    for epoca in range(EPOCHE):
        modello.train()
        perdita_totale_train = 0.0
        for X_batch, y_batch in loader_train:
            X_batch = X_batch.to(dispositivo)
            y_batch = y_batch.to(dispositivo)
            predizioni = modello(X_batch)
            loss = criterio_loss(predizioni, y_batch)
            ottimizzatore.zero_grad()
            loss.backward()
            ottimizzatore.step()
            perdita_totale_train += loss.item()

        loss_media_train = perdita_totale_train / len(loader_train)

        modello.eval()
        perdita_totale_val = 0.0
        corrette = 0
        totali = 0
        with torch.no_grad():
            for X_batch, y_batch in loader_val:
                X_batch = X_batch.to(dispositivo)
                y_batch = y_batch.to(dispositivo)
                predizioni = modello(X_batch)
                loss = criterio_loss(predizioni, y_batch)
                perdita_totale_val += loss.item()
                classi_predette = predizioni.argmax(1)
                corrette += (classi_predette == y_batch).sum().item()
                totali += y_batch.size(0)

        loss_media_val = perdita_totale_val / len(loader_val)
        accuracy_val = corrette / totali

        storico_loss_train.append(loss_media_train)
        storico_loss_val.append(loss_media_val)
        storico_accuracy_val.append(accuracy_val)

        print(f"Epoca [{epoca+1:3d}/{EPOCHE}]  "
              f"Train Loss: {loss_media_train:.4f}  "
              f"Val Loss: {loss_media_val:.4f}  "
              f"Val Accuracy: {accuracy_val:.4f}")

        if loss_media_val < miglior_val_loss:
            miglior_val_loss = loss_media_val
            contatore_pazienza = 0
            torch.save(modello.state_dict(), percorso_modello)
            print(f"  ✓ Modello salvato (miglior val_loss: {miglior_val_loss:.4f})")
        else:
            contatore_pazienza += 1
            print(f"  ✗ Nessun miglioramento ({contatore_pazienza}/{PAZIENZA})")
            if contatore_pazienza >= PAZIENZA:
                print(f"\n⚠ Early stopping!")
                break

    print(f"\n{'='*60}")
    print(f" Addestramento completato — {epoca+1} epoche")
    print(f" Modello salvato in: {percorso_modello}")
    print(f"{'='*60}")

    modello.load_state_dict(torch.load(percorso_modello, map_location=dispositivo))
    modello.eval()
    tutte_predizioni = []
    tutte_label = []
    with torch.no_grad():
        for X_batch, y_batch in loader_val:
            X_batch = X_batch.to(dispositivo)
            predizioni = modello(X_batch)
            tutte_predizioni.extend(predizioni.argmax(1).cpu().numpy())
            tutte_label.extend(y_batch.numpy())

    tutte_predizioni = np.array(tutte_predizioni)
    tutte_label = np.array(tutte_label)

    print(f"\n{'='*60}")
    print(" METRICHE DI VALUTAZIONE")
    print(f"{'='*60}\n")
    print(classification_report(tutte_label, tutte_predizioni,
                                target_names=["no_fight", "fight"]))
    cm = confusion_matrix(tutte_label, tutte_predizioni)
    print("Matrice di confusione:")
    print(cm)

    # GRAFICI
    os.makedirs("grafici", exist_ok=True)
    epoche_range = range(1, len(storico_loss_train) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epoche_range, storico_loss_train, label="Train Loss", marker='o', markersize=3)
    plt.plot(epoche_range, storico_loss_val, label="Val Loss", marker='o', markersize=3)
    plt.xlabel("Epoca"); plt.ylabel("Loss")
    plt.title("Loss — Coordinate Relative")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("grafici/loss_relativo.png", dpi=150); plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(epoche_range, storico_accuracy_val, label="Val Accuracy", marker='o',
             markersize=3, color='green')
    plt.xlabel("Epoca"); plt.ylabel("Accuracy")
    plt.title("Accuracy — Coordinate Relative")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("grafici/accuracy_relativo.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["no_fight", "fight"])
    ax.set_yticklabels(["no_fight", "fight"])
    ax.set_xlabel("Predetto"); ax.set_ylabel("Reale")
    ax.set_title("Matrice di Confusione — Coordinate Relative")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=20, fontweight='bold')
    plt.colorbar(im); plt.tight_layout()
    plt.savefig("grafici/confusion_matrix_relativo.png", dpi=150); plt.close()

    print("\nGrafici salvati in grafici/")
    print(f"\n{'='*60}")
    print(f" Modello:  {percorso_modello}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
