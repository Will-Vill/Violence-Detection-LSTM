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
from sklearn.preprocessing import StandardScaler
import random
from collections import defaultdict

# Fissa il seed per rendere l'addestramento deterministico e riproducibile
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

# Seleziona hardware
dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Viene utilizzato il dispositivo: {dispositivo}")

CARTELLA_INPUT = "output"
CARTELLA_MODELLI = "modelli"

# Costanti e iperparametri della rete
FINESTRA = 30
STRIDE = 15

# FEATURES MULTI-PERSONA (per frame):
#   Persona A:  34 coordinate + 34 velocità = 68
#   Persona B:  34 coordinate + 34 velocità = 68
#   Distanza tra i centri:      1
#   Variazione della distanza:  1
#   Numero persone nel frame:   1
#   TOTALE:                     139
NUM_FEATURES = 139

HIDDEN_1 = 192   # Aumentato per gestire 139 features
HIDDEN_2 = 64
DENSE_1 = 32
DENSE_2 = 16
DROPOUT = 0.4    # Aumentato per contrastare l'overfitting
NUM_CLASSI = 2
LEARNING_RATE = 0.0005  # Ridotto per maggiore stabilità
BATCH_SIZE = 32
EPOCHE = 50
PAZIENZA = 10

NUM_KP = 17
KP_SIZE = NUM_KP * 2  # 34 valori (17 keypoints x 2 coordinate)


# ══════════════════════════════════════════════════════════════════════
# PREPARAZIONE DATI — VERSIONE MULTI-PERSONA (COPPIA STABILE)
# ══════════════════════════════════════════════════════════════════════

def trova_coppia_stabile(df):
    """
    Trova la coppia di persone che appare più vicina IN MEDIA
    nell'intero video. Questa coppia viene poi usata per tutti
    i frame, garantendo coerenza temporale nella sliding window.

    Ritorna (id_A, id_B) oppure None se non ci sono coppie.
    """
    # Calcola il centro di ogni persona per ogni frame
    centri_per_frame = {}
    for frame_num, dati_frame in df.groupby('frame'):
        centri = {}
        for _, riga in dati_frame.iterrows():
            pid = int(riga['id_persona'])
            coords = riga.iloc[2:].values.astype(float)
            x_coords = coords[0::2]
            y_coords = coords[1::2]
            centri[pid] = np.array([np.mean(x_coords), np.mean(y_coords)])
        centri_per_frame[frame_num] = centri

    # Per ogni coppia di persone, calcola la distanza media
    # su tutti i frame in cui appaiono ENTRAMBE
    distanze_coppie = defaultdict(list)

    for frame_num, centri in centri_per_frame.items():
        ids = list(centri.keys())
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                coppia = (min(ids[i], ids[j]), max(ids[i], ids[j]))
                dist = np.sqrt(np.sum((centri[ids[i]] - centri[ids[j]]) ** 2))
                distanze_coppie[coppia].append(dist)

    if not distanze_coppie:
        return None

    # Trova la coppia con la distanza media minima che appare
    # in almeno 15 frame (mezzo secondo) per essere significativa
    migliore_coppia = None
    migliore_dist = float('inf')

    for coppia, distanze in distanze_coppie.items():
        if len(distanze) >= 15:  # Almeno mezzo secondo di co-presenza
            dist_media = np.mean(distanze)
            if dist_media < migliore_dist:
                migliore_dist = dist_media
                migliore_coppia = coppia

    # Se nessuna coppia ha 15+ frame, prendi quella con più frame
    if migliore_coppia is None:
        migliore_coppia = max(distanze_coppie.keys(),
                              key=lambda c: len(distanze_coppie[c]))

    return migliore_coppia


def crea_sequenze_da_csv(percorso_csv):
    """
    Versione multi-persona con COPPIA STABILE:
    1. Trova la coppia di persone più vicina in media nel video
    2. Usa SEMPRE quelle due persone per costruire le features
    3. La sliding window scorre su frame coerenti (stessa coppia)
    """
    df = pd.read_csv(percorso_csv)

    if df.empty:
        return []

    tutti_frame = sorted(df['frame'].unique())
    if len(tutti_frame) < 2:
        return []

    # --- PASSO 1: Trova la coppia stabile ---
    coppia = trova_coppia_stabile(df)
    if coppia is None:
        return []

    id_A, id_B = coppia

    # --- PASSO 2: Costruisci le features per la coppia stabile ---
    frame_min = tutti_frame[0]
    frame_max = tutti_frame[-1]
    frame_range = list(range(frame_min, frame_max + 1))

    # Pre-calcola i dati per frame
    dati_per_frame = {}
    centri_per_frame = {}
    conteggio_per_frame = {}

    for frame_num, dati_frame in df.groupby('frame'):
        persone = {}
        centri = {}
        for _, riga in dati_frame.iterrows():
            pid = int(riga['id_persona'])
            coords = riga.iloc[2:].values.astype(float)
            persone[pid] = coords
            x_coords = coords[0::2]
            y_coords = coords[1::2]
            centri[pid] = np.array([np.mean(x_coords), np.mean(y_coords)])
        dati_per_frame[frame_num] = persone
        centri_per_frame[frame_num] = centri
        conteggio_per_frame[frame_num] = len(persone)

    lista_coord_A = []
    lista_coord_B = []
    lista_distanze = []
    lista_num_persone = []

    # Ultimo valore valido per riempire i buchi
    ultimo_A = np.zeros(KP_SIZE)
    ultimo_B = np.zeros(KP_SIZE)

    for frame_num in frame_range:
        persone = dati_per_frame.get(frame_num, {})
        centri = centri_per_frame.get(frame_num, {})
        n_persone = conteggio_per_frame.get(frame_num, 0)

        # Prendi le coordinate della coppia stabile
        coord_A = persone.get(id_A, None)
        coord_B = persone.get(id_B, None)

        if coord_A is not None:
            ultimo_A = coord_A
        if coord_B is not None:
            ultimo_B = coord_B

        # Usa l'ultimo valore valido se manca in questo frame
        lista_coord_A.append(ultimo_A.copy())
        lista_coord_B.append(ultimo_B.copy())

        # Calcola distanza tra i centri della coppia
        if coord_A is not None and coord_B is not None:
            centro_A = centri[id_A]
            centro_B = centri[id_B]
            dist = np.sqrt(np.sum((centro_A - centro_B) ** 2))
            lista_distanze.append(dist)
        else:
            # Se uno dei due non è visibile, usa distanza neutra
            lista_distanze.append(0.5)

        lista_num_persone.append(n_persone)

    # Converti in array numpy
    coord_A = np.array(lista_coord_A)       # (T, 34)
    coord_B = np.array(lista_coord_B)       # (T, 34)
    distanze = np.array(lista_distanze).reshape(-1, 1)
    num_persone = np.array(lista_num_persone).reshape(-1, 1)

    if len(coord_A) < 2:
        return []

    # --- PASSO 3: Velocità ---
    vel_A = np.diff(coord_A, axis=0)
    vel_A = np.vstack([np.zeros((1, KP_SIZE)), vel_A])

    vel_B = np.diff(coord_B, axis=0)
    vel_B = np.vstack([np.zeros((1, KP_SIZE)), vel_B])

    # --- PASSO 4: Delta distanza ---
    delta_dist = np.diff(distanze, axis=0)
    delta_dist = np.vstack([np.zeros((1, 1)), delta_dist])

    # --- PASSO 5: Concatena ---
    features = np.hstack([
        coord_A, coord_B,
        vel_A, vel_B,
        distanze, delta_dist,
        num_persone
    ])

    # --- PASSO 6: Sliding window ---
    sequenze_video = []
    for i in range(0, len(features) - FINESTRA + 1, STRIDE):
        fetta_video = features[i : i + FINESTRA]
        sequenze_video.append(fetta_video)

    return sequenze_video



def carica_dataset(split):
    """
    Cerca le cartelle (train/val) e costruisce i dataset finali,
    associa l'etichetta 1 per fight e 0 per no_fight.
    """
    X = []
    y = []

    categorie = ["fight", "no_fight"]

    for categoria in categorie:
        etichetta = 1 if categoria == "fight" else 0
        cartella = os.path.join(CARTELLA_INPUT, split, categoria)
        lista_file_csv = glob.glob(os.path.join(cartella, "*.csv"))

        for file_csv in lista_file_csv:
            sequenze = crea_sequenze_da_csv(file_csv)
            for sequenza in sequenze:
                X.append(sequenza)
                y.append(etichetta)

        print(f"  {categoria}: {sum(1 for label in y if label == etichetta)} sequenze totali")

    return np.array(X), np.array(y)



# CLASSI PYTORCH
class FightDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



class LSTMClassificatore(nn.Module):
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
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout4(x)
        x = self.fc3(x)
        return x


def main():
    # CARICAMENTO DATI
    print("Caricamento Train Set...")
    X_train, y_train = carica_dataset("train")

    print("\nCaricamento Validation Set...")
    X_val, y_val = carica_dataset("val")

    if len(X_train) == 0 or len(X_val) == 0:
        print("ERRORE: Non ci sono abbastanza dati per addestrare il modello")
        return

    # --- NORMALIZZAZIONE con StandardScaler ---
    # Reshape per lo scaler: (N*30, 139)
    n_train = X_train.shape[0]
    n_val = X_val.shape[0]

    X_train_flat = X_train.reshape(-1, NUM_FEATURES)
    X_val_flat = X_val.reshape(-1, NUM_FEATURES)

    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_val_flat = scaler.transform(X_val_flat)

    X_train = X_train_flat.reshape(n_train, FINESTRA, NUM_FEATURES)
    X_val = X_val_flat.reshape(n_val, FINESTRA, NUM_FEATURES)

    print(f"\nX_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}    y_val:   {y_val.shape}")
    print(f"Distribuzione train → fight: {sum(y_train==1)}, no_fight: {sum(y_train==0)}")
    print(f"Distribuzione val   → fight: {sum(y_val==1)}, no_fight: {sum(y_val==0)}")

    dataset_train = FightDataset(X_train, y_train)
    dataset_val = FightDataset(X_val, y_val)

    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)

    modello = LSTMClassificatore().to(dispositivo)

    criterio_loss = nn.CrossEntropyLoss()
    ottimizzatore = optim.Adam(modello.parameters(), lr=LEARNING_RATE)

    # ReduceLROnPlateau: riduce il LR quando la val_loss smette di migliorare
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        ottimizzatore, mode='min', factor=0.5, patience=5, verbose=True
    )

    # TRAINING
    storico_loss_train = []
    storico_loss_val = []
    storico_accuracy_val = []

    miglior_val_loss = float('inf')
    contatore_pazienza = 0

    os.makedirs(CARTELLA_MODELLI, exist_ok=True)
    percorso_modello = os.path.join(CARTELLA_MODELLI, "lstm_risse_multi.pt")

    print(f"\n{'='*60}")
    print(f" Inizio addestramento MULTI-PERSONA — max {EPOCHE} epoche")
    print(f"{'='*60}\n")

    for epoca in range(EPOCHE):

        # FASE TRAINING
        modello.train()
        perdita_totale_train = 0.0

        for X_batch, y_batch in loader_train:
            X_batch = X_batch.to(dispositivo)
            y_batch = y_batch.to(dispositivo)

            predizioni = modello(X_batch)
            loss = criterio_loss(predizioni, y_batch)

            ottimizzatore.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modello.parameters(), max_norm=1.0)
            ottimizzatore.step()

            perdita_totale_train += loss.item()

        loss_media_train = perdita_totale_train / len(loader_train)

        # FASE VALIDAZIONE
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

        # Aggiorna lo scheduler
        scheduler.step(loss_media_val)

        storico_loss_train.append(loss_media_train)
        storico_loss_val.append(loss_media_val)
        storico_accuracy_val.append(accuracy_val)

        print(f"Epoca [{epoca+1:3d}/{EPOCHE}]  "
              f"Train Loss: {loss_media_train:.4f}  "
              f"Val Loss: {loss_media_val:.4f}  "
              f"Val Accuracy: {accuracy_val:.4f}")

        # EARLY STOPPING
        if loss_media_val < miglior_val_loss:
            miglior_val_loss = loss_media_val
            contatore_pazienza = 0
            torch.save(modello.state_dict(), percorso_modello)
            print(f"  ✓ Modello salvato (miglior val_loss: {miglior_val_loss:.4f})")
        else:
            contatore_pazienza += 1
            print(f"  ✗ Nessun miglioramento ({contatore_pazienza}/{PAZIENZA})")
            if contatore_pazienza >= PAZIENZA:
                print(f"\n⚠ Early stopping! La val_loss non migliora da {PAZIENZA} epoche.")
                break

    print(f"\n{'='*60}")
    print(f" Addestramento completato — {epoca+1} epoche")
    print(f" Modello salvato in: {percorso_modello}")
    print(f"{'='*60}")


    # VALUTAZIONI FINALI
    modello.load_state_dict(torch.load(percorso_modello, map_location=dispositivo))
    modello.eval()

    tutte_predizioni = []
    tutte_label = []

    with torch.no_grad():
        for X_batch, y_batch in loader_val:
            X_batch = X_batch.to(dispositivo)
            predizioni = modello(X_batch)
            classi_predette = predizioni.argmax(1).cpu().numpy()
            tutte_predizioni.extend(classi_predette)
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
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.title("Andamento della Loss — Modello Multi-Persona")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("grafici/loss_multi.png", dpi=150)
    plt.close()
    print("\nGrafico salvato: grafici/loss_multi.png")

    plt.figure(figsize=(10, 5))
    plt.plot(epoche_range, storico_accuracy_val, label="Val Accuracy", marker='o',
             markersize=3, color='green')
    plt.xlabel("Epoca")
    plt.ylabel("Accuracy")
    plt.title("Andamento dell'Accuracy — Modello Multi-Persona")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("grafici/accuracy_multi.png", dpi=150)
    plt.close()
    print("Grafico salvato: grafici/accuracy_multi.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["no_fight", "fight"])
    ax.set_yticklabels(["no_fight", "fight"])
    ax.set_xlabel("Predetto")
    ax.set_ylabel("Reale")
    ax.set_title("Matrice di Confusione — Multi-Persona")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=20, fontweight='bold')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("grafici/confusion_matrix_multi.png", dpi=150)
    plt.close()
    print("Grafico salvato: grafici/confusion_matrix_multi.png")

    print(f"\n{'='*60}")
    print(" TUTTO COMPLETATO!")
    print(f" Modello:  {percorso_modello}")
    print(f" Grafici:  grafici/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
