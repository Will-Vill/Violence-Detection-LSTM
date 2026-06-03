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
NUM_FEATURES = 71  # 34 coord + 34 vel + 1 dist + 1 delta_dist + 1 num_persone
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

# Soglia filtro spettatori
SOGLIA_COMPAGNIA = 0.3


def crea_sequenze_da_csv(percorso_csv):
    """
    Per-persona (groupby id_persona) + filtro spettatori + features arricchite.
    Struttura: coord(34) + vel(34) + dist(1) + delta_dist(1) + num_persone(1) = 71
    """
    df = pd.read_csv(percorso_csv)
    if df.empty:
        return []

    sequenze_video = []

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
        coordinate = dati_persona.iloc[:, 2:].values
        lista_frame = dati_persona['frame'].values

        if len(coordinate) < 2:
            continue

        velocita = np.diff(coordinate, axis=0)
        velocita = np.vstack([np.zeros((1, coordinate.shape[1])), velocita])

        distanze = []
        num_persone = []

        for frame_num in lista_frame:
            centri = centri_per_frame.get(frame_num, {})
            num_persone.append(len(centri))

            if len(centri) < 2 or id_persona not in centri:
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
        num_persone = np.array(num_persone).reshape(-1, 1)

        delta_dist = np.diff(distanze, axis=0)
        delta_dist = np.vstack([np.zeros((1, 1)), delta_dist])

        features = np.hstack([coordinate, velocita, distanze, delta_dist, num_persone])

        for i in range(0, len(features) - FINESTRA + 1, STRIDE):
            fetta = features[i : i + FINESTRA]

            # Filtro spettatori
            dist_finestra = fetta[:, 68]
            frame_con_vicino = np.sum(dist_finestra < 1.0)
            if frame_con_vicino / FINESTRA < SOGLIA_COMPAGNIA:
                continue

            sequenze_video.append(fetta)

    return sequenze_video


def carica_dataset(split):
    X = []
    y = []
    for categoria in ["fight", "no_fight"]:
        etichetta = 1 if categoria == "fight" else 0
        cartella = os.path.join(CARTELLA_INPUT, split, categoria)
        lista_file_csv = glob.glob(os.path.join(cartella, "*.csv"))
        for file_csv in lista_file_csv:
            sequenze = crea_sequenze_da_csv(file_csv)
            print(f"  {categoria}: {len(sequenze)} sequenze da {file_csv}")
            for sequenza in sequenze:
                X.append(sequenza)
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


class TemporalAttention(nn.Module):
    """
    Attenzione temporale: invece di prendere solo l'ultimo frame,
    calcola un peso per ogni frame della sequenza e fa una media pesata.
    Così la rete impara QUALI frame sono più importanti per la classificazione.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch, 30, hidden_size)
        scores = self.attention(lstm_output)          # (batch, 30, 1)
        weights = torch.softmax(scores, dim=1)        # (batch, 30, 1)
        context = torch.sum(lstm_output * weights, dim=1)  # (batch, hidden_size)
        return context


class LSTMClassificatore(nn.Module):
    """
    Architettura migliorata:
    - BiLSTM: legge la sequenza in avanti E all'indietro
    - Attenzione temporale: impara quali frame contano di più
    """
    def __init__(self):
        super(LSTMClassificatore, self).__init__()
        self.lstm1 = nn.LSTM(input_size=NUM_FEATURES, hidden_size=HIDDEN_1,
                             batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(DROPOUT)
        self.lstm2 = nn.LSTM(input_size=HIDDEN_1 * 2, hidden_size=HIDDEN_2,
                             batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(DROPOUT)

        # Attenzione temporale al posto di x[:, -1, :]
        self.attention = TemporalAttention(HIDDEN_2 * 2)

        self.fc1 = nn.Linear(HIDDEN_2 * 2, DENSE_1)
        self.dropout3 = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(DENSE_1, DENSE_2)
        self.dropout4 = nn.Dropout(DROPOUT)
        self.fc3 = nn.Linear(DENSE_2, NUM_CLASSI)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        # Attenzione: media pesata di TUTTI i frame (non solo l'ultimo)
        x = self.attention(x)

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

    # NIENTE StandardScaler — il base senza scaler addestrava per 36 epoche
    # senza overfitting. Con lo scaler la convergenza era troppo veloce.

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
    percorso_modello = os.path.join(CARTELLA_MODELLI, "lstm_risse_multiv7.pt")

    print(f"\n{'='*60}")
    print(f" Inizio addestramento v7 (BiLSTM+Attention) — max {EPOCHE} epoche")
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
                print(f"\n⚠ Early stopping! La val_loss non migliora da {PAZIENZA} epoche.")
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
    plt.title("Andamento della Loss — v7 BiLSTM+Attention")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("grafici/loss_multiv7.png", dpi=150)
    plt.close()
    print("\nGrafico salvato: grafici/loss_multiv7.png")

    plt.figure(figsize=(10, 5))
    plt.plot(epoche_range, storico_accuracy_val, label="Val Accuracy", marker='o',
             markersize=3, color='green')
    plt.xlabel("Epoca")
    plt.ylabel("Accuracy")
    plt.title("Andamento dell'Accuracy — v7 BiLSTM+Attention")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("grafici/accuracy_multiv7.png", dpi=150)
    plt.close()
    print("Grafico salvato: grafici/accuracy_multiv7.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["no_fight", "fight"])
    ax.set_yticklabels(["no_fight", "fight"])
    ax.set_xlabel("Predetto")
    ax.set_ylabel("Reale")
    ax.set_title("Matrice di Confusione — v7")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=20, fontweight='bold')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("grafici/confusion_matrix_multiv7.png", dpi=150)
    plt.close()
    print("Grafico salvato: grafici/confusion_matrix_multiv7.png")

    print(f"\n{'='*60}")
    print(" TUTTO COMPLETATO!")
    print(f" Modello:  {percorso_modello}")
    print(f" Grafici:  grafici/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
