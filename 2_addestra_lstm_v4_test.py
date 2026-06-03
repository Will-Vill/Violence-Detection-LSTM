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

CARTELLA_INPUT = "tipi_output_yolo/output_YOLOnNANO"
CARTELLA_MODELLI = "modello"

# Costanti e iperparametri — IDENTICI al modello base
FINESTRA = 30
STRIDE = 15
NUM_FEATURES = 104  # 34 coordinate + 34 velocità + 1 distanza
HIDDEN_1 = 128
HIDDEN_2 = 64
DENSE_1 = 32
DENSE_2 = 16
DROPOUT = 0.45
NUM_CLASSI = 2
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHE = 50
PAZIENZA = 10


# ══════════════════════════════════════════════════════════════════════
# PREPARAZIONE DATI — UGUALE AL BASE + FILTRO SPETTATORI
# ══════════════════════════════════════════════════════════════════════

def crea_sequenze_da_csv(percorso_csv, categoria):
    df = pd.read_csv(percorso_csv)
    sequenze_video = []

    # --- PASSO 1: Centro di massa per il calcolo distanze ---
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

    # --- FILTRO SPETTATORI (Solo nei video Fight) ---
    persone_da_tenere = set(df['id_persona'].unique())

    if categoria == "fight":
        velocita_per_persona = {}
        for id_persona, dati_persona in df.groupby('id_persona'):
            coords = dati_persona.iloc[:, 2:].values.astype(float)
            if len(coords) < 2:
                velocita_per_persona[id_persona] = 0.0
                continue
            diff = np.diff(coords, axis=0)
            vel_media = np.mean(np.abs(diff))
            velocita_per_persona[id_persona] = vel_media

        if len(velocita_per_persona) > 2:
            valori_vel = list(velocita_per_persona.values())
            mediana_vel = np.median(valori_vel)
            persone_da_tenere = {pid for pid, vel in velocita_per_persona.items() if vel >= mediana_vel}
            if len(persone_da_tenere) < 2:
                top2 = sorted(velocita_per_persona.keys(), key=lambda p: velocita_per_persona[p], reverse=True)[:2]
                persone_da_tenere = set(top2)

    # --- PASSO 2: Feature Engineering Potenziato ---
    for id_persona, dati_persona in df.groupby('id_persona'):
        if id_persona not in persone_da_tenere:
            continue

        coordinate = dati_persona.iloc[:, 2:].values
        lista_frame = dati_persona['frame'].values

        if len(coordinate) < 3: # Serve almeno lunghezza 3 per l'accelerazione
            continue

        # 1. VELOCITÀ E ACCELERAZIONE (Basate sulle coordinate assolute)
        velocita = np.diff(coordinate, axis=0)
        velocita = np.vstack([np.zeros((1, coordinate.shape[1])), velocita])
        
        accelerazione = np.diff(velocita, axis=0)
        accelerazione = np.vstack([np.zeros((1, velocita.shape[1])), accelerazione])

        # 2. CENTRATURA DELLO SCHELETRO (Invarianza Traslazionale)
        # Sottraiamo il centro di massa della persona alle sue stesse coordinate
        centri_x = np.mean(coordinate[:, 0::2], axis=1, keepdims=True)
        centri_y = np.mean(coordinate[:, 1::2], axis=1, keepdims=True)
        
        coord_centrate = coordinate.copy()
        coord_centrate[:, 0::2] = coordinate[:, 0::2] - centri_x
        coord_centrate[:, 1::2] = coordinate[:, 1::2] - centri_y

        # 3. DISTANZA RELAZIONALE
        distanze = []
        for frame_num in lista_frame:
            centri = centri_per_frame.get(frame_num, {})
            if len(centri) < 2:
                distanze.append(1.0)
            else:
                centro_persona = centri[id_persona]
                min_dist = float('inf')
                for altro_id, altro_centro in centri.items():
                    if altro_id != id_persona:
                        dist = np.sqrt(np.sum((centro_persona - altro_centro) ** 2))
                        if dist < min_dist:
                            min_dist = dist
                distanze.append(min_dist)

        distanze = np.array(distanze).reshape(-1, 1)

        # 4. DELTA DISTANZA (Si stanno avvicinando o allontanando?)
        delta_dist = np.diff(distanze, axis=0)
        delta_dist = np.vstack([np.zeros((1, 1)), delta_dist])

        # Concatena tutto: coord_centrate(34) + vel(34) + acc(34) + dist(1) + delta_dist(1) = 104
        features = np.hstack([coord_centrate, velocita, accelerazione, distanze, delta_dist])

        # Sliding window
        for i in range(0, len(features) - FINESTRA + 1, STRIDE):
            fetta_video = features[i : i + FINESTRA]
            sequenze_video.append(fetta_video)

    return sequenze_video



def carica_dataset(split):
    X = []
    y = []

    categorie = ["fight", "no_fight"]

    for categoria in categorie:
        etichetta = 1 if categoria == "fight" else 0
        cartella = os.path.join(CARTELLA_INPUT, split, categoria)
        lista_file_csv = glob.glob(os.path.join(cartella, "*.csv"))

        for file_csv in lista_file_csv:
            # Passa la categoria per attivare il filtro solo sui fight
            sequenze = crea_sequenze_da_csv(file_csv, categoria)
            for sequenza in sequenze:
                X.append(sequenza)
                y.append(etichetta)

        n_cat = sum(1 for label in y if label == etichetta)
        print(f"  {categoria}: {n_cat} sequenze totali")

    return np.array(X), np.array(y)



# CLASSI PYTORCH — IDENTICHE AL BASE
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
    Architettura IDENTICA al modello base.
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

    pesi = torch.tensor([14362/21248, 1.0], dtype=torch.float32).to(dispositivo)
    criterio_loss = nn.CrossEntropyLoss(weight=pesi)
    ottimizzatore = optim.Adam(modello.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        ottimizzatore, mode='min', factor=0.5, patience=5
    )

    # TRAINING
    storico_loss_train = []
    storico_loss_val = []
    storico_accuracy_val = []

    miglior_val_loss = float('inf')
    contatore_pazienza = 0

    os.makedirs(CARTELLA_MODELLI, exist_ok=True)
    percorso_modello = os.path.join(CARTELLA_MODELLI, "lstm_risse.pt")

    print(f"\n{'='*60}")
    print(f" Inizio addestramento — max {EPOCHE} epoche")
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
    plt.title("Andamento della Loss durante il Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("grafici/loss.png", dpi=150)
    plt.close()
    print("\nGrafico salvato: grafici/loss.png")

    plt.figure(figsize=(10, 5))
    plt.plot(epoche_range, storico_accuracy_val, label="Val Accuracy", marker='o',
             markersize=3, color='green')
    plt.xlabel("Epoca")
    plt.ylabel("Accuracy")
    plt.title("Andamento dell'Accuracy di Validazione")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("grafici/accuracy.png", dpi=150)
    plt.close()
    print("Grafico salvato: grafici/accuracy.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["no_fight", "fight"])
    ax.set_yticklabels(["no_fight", "fight"])
    ax.set_xlabel("Predetto")
    ax.set_ylabel("Reale")
    ax.set_title("Matrice di Confusione")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=20, fontweight='bold')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("grafici/confusion_matrix.png", dpi=150)
    plt.close()
    print("Grafico salvato: grafici/confusion_matrix.png")

    print(f"\n{'='*60}")
    print(" TUTTO COMPLETATO!")
    print(f" Modello:  {percorso_modello}")
    print(f" Grafici:  grafici/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
