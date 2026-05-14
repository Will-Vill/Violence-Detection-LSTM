import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Seleziona hardware
dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Viene utilizzato il dispositivo: {dispositivo}")

CARTELLA_INPUT = "output"
CARTELLA_MODELLI = "modelli"

# ===========================================================================
# IPERPARAMETRI
# ===========================================================================
FINESTRA = 30
STRIDE = 15
MAX_PERSONE = 2
# 12 keypoints corpo (esclusi viso: naso, occhi, orecchie)
KP_CORPO = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
NUM_KP = len(KP_CORPO)
COORD_PER_KP = 2

# Features per frame:
#   48 coordinate (2 persone × 12 kp × 2)
# + 48 velocità
# +  1 distanza tra le 2 persone
# +  1 numero persone nel frame
# = 98
NUM_FEATURES = MAX_PERSONE * NUM_KP * COORD_PER_KP * 2 + 2  # 98

HIDDEN_1 = 128
HIDDEN_2 = 64
DENSE_1 = 32
DROPOUT = 0.3
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHE = 80
PAZIENZA = 15


# ===========================================================================
# PREPARAZIONE DATI
# ===========================================================================
def trova_coppia_piu_vicina(keypoints_frame):
    """
    Dato un dict {id_persona: array_keypoints_corpo}, trova la coppia
    di persone più vicina e restituisce i loro keypoints.
    Se c'è solo 1 persona, restituisce quella + zeri.
    Se non ci sono persone, restituisce zeri.
    """
    kp_size = NUM_KP * COORD_PER_KP  # 24
    p0 = np.zeros(kp_size)
    p1 = np.zeros(kp_size)
    distanza = 0.0
    num_persone = len(keypoints_frame)

    if num_persone == 0:
        return p0, p1, distanza, 0

    ids = list(keypoints_frame.keys())

    if num_persone == 1:
        p0 = keypoints_frame[ids[0]]
        return p0, p1, distanza, 1

    # Trova la coppia con distanza minima (centro di massa)
    min_dist = float('inf')
    best_i, best_j = 0, 1

    for i in range(len(ids)):
        kp_i = keypoints_frame[ids[i]].reshape(NUM_KP, 2)
        centro_i = np.mean(kp_i, axis=0)
        for j in range(i + 1, len(ids)):
            kp_j = keypoints_frame[ids[j]].reshape(NUM_KP, 2)
            centro_j = np.mean(kp_j, axis=0)
            dist = np.sqrt(np.sum((centro_i - centro_j) ** 2))
            if dist < min_dist:
                min_dist = dist
                best_i, best_j = i, j

    p0 = keypoints_frame[ids[best_i]]
    p1 = keypoints_frame[ids[best_j]]
    distanza = min_dist

    return p0, p1, distanza, min(num_persone, MAX_PERSONE)


def crea_sequenze_da_csv(percorso_csv):
    """
    Legge un CSV. Per ogni frame, trova la coppia di persone più vicina.
    Calcola velocità e distanza. Crea sequenze con sliding window.
    """
    df = pd.read_csv(percorso_csv)
    if df.empty:
        return []

    colonne_kp = [c for c in df.columns if c not in ('frame', 'id_persona')]

    # Indici colonne dei 12 keypoints corpo
    colonne_corpo = []
    for idx_kp in KP_CORPO:
        colonne_corpo.append(colonne_kp[idx_kp * 2])
        colonne_corpo.append(colonne_kp[idx_kp * 2 + 1])

    # Costruisci feature frame per frame
    tutti_frame = sorted(df['frame'].unique())
    # Aggiungi anche i frame dove nessuno è stato rilevato
    if tutti_frame:
        tutti_frame_completi = list(range(tutti_frame[0], tutti_frame[-1] + 1))
    else:
        return []

    matrice_coord = []  # coordinate delle 2 persone più vicine
    distanze = []
    num_persone_list = []

    for frame_num in tutti_frame_completi:
        dati_frame = df[df['frame'] == frame_num]

        # Raccogli keypoints per ogni persona in questo frame
        kp_per_persona = {}
        if len(dati_frame) > 0:
            for _, riga in dati_frame.iterrows():
                pid = int(riga['id_persona'])
                coords = riga[colonne_corpo].values.astype(float)
                # Salta persone con troppi keypoints a zero (non rilevate bene)
                if np.count_nonzero(coords) > len(coords) * 0.5:
                    kp_per_persona[pid] = coords

        p0, p1, dist, n_pers = trova_coppia_piu_vicina(kp_per_persona)
        matrice_coord.append(np.concatenate([p0, p1]))  # 48 valori
        distanze.append(dist)
        num_persone_list.append(n_pers)

    matrice_coord = np.array(matrice_coord)
    distanze = np.array(distanze).reshape(-1, 1)
    num_persone_arr = np.array(num_persone_list).reshape(-1, 1)

    if len(matrice_coord) < 2:
        return []

    # Velocità (differenza tra frame consecutivi)
    velocita = np.diff(matrice_coord, axis=0)
    velocita = np.vstack([np.zeros((1, velocita.shape[1])), velocita])

    # Combina tutto: coord(48) + velocità(48) + distanza(1) + num_persone(1) = 98
    features = np.hstack([matrice_coord, velocita, distanze, num_persone_arr])

    # Sliding window
    sequenze = []
    for i in range(0, len(features) - FINESTRA + 1, STRIDE):
        sequenze.append(features[i: i + FINESTRA])

    return sequenze


def carica_dataset(split):
    """Carica tutti i CSV di un split e crea X, y."""
    X = []
    y = []

    for categoria in ["fight", "no_fight"]:
        etichetta = 1 if categoria == "fight" else 0
        cartella = os.path.join(CARTELLA_INPUT, split, categoria)
        lista_csv = glob.glob(os.path.join(cartella, "*.csv"))

        contatore = 0
        for file_csv in lista_csv:
            sequenze = crea_sequenze_da_csv(file_csv)
            for seq in sequenze:
                X.append(seq)
                y.append(etichetta)
            contatore += len(sequenze)

        print(f"  {categoria}: {contatore} sequenze da {len(lista_csv)} video")

    return np.array(X, dtype=np.float32), np.array(y)


# ===========================================================================
# RETE NEURALE
# ===========================================================================
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
    LSTM bidirezionale a 2 livelli con normalizzazione.
    """
    def __init__(self):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=NUM_FEATURES, hidden_size=HIDDEN_1,
                             batch_first=True, bidirectional=True)
        self.norm1 = nn.LayerNorm(HIDDEN_1 * 2)
        self.dropout1 = nn.Dropout(DROPOUT)

        self.lstm2 = nn.LSTM(input_size=HIDDEN_1 * 2, hidden_size=HIDDEN_2,
                             batch_first=True, bidirectional=True)
        self.norm2 = nn.LayerNorm(HIDDEN_2 * 2)
        self.dropout2 = nn.Dropout(DROPOUT)

        self.fc1 = nn.Linear(HIDDEN_2 * 2, DENSE_1)
        self.dropout3 = nn.Dropout(DROPOUT)
        self.fc_out = nn.Linear(DENSE_1, 2)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.norm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.norm2(x)
        x = self.dropout2(x)

        x = x[:, -1, :]  # Ultimo timestep
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc_out(x)
        return x


# ===========================================================================
# TRAINING
# ===========================================================================
def main():
    print("=" * 60)
    print(" ADDESTRAMENTO LSTM — Riconoscimento Risse v3")
    print("=" * 60)
    print(f"\nMiglioramenti rispetto a v2:")
    print(f"  - Coppia più vicina per frame (non per ID più frequente)")
    print(f"  - LSTM bidirezionale + LayerNorm")
    print(f"  - StandardScaler sui dati")
    print(f"  - Learning rate scheduler + gradient clipping")
    print(f"  - Features: {NUM_FEATURES} per frame")

    # Caricamento dati
    print(f"\nCaricamento Train Set...")
    X_train, y_train = carica_dataset("train")
    print(f"\nCaricamento Validation Set...")
    X_val, y_val = carica_dataset("val")

    if len(X_train) == 0 or len(X_val) == 0:
        print("ERRORE: Non ci sono abbastanza dati")
        return

    # --- Normalizzazione con StandardScaler ---
    # Reshaping: (N, 30, 98) → (N*30, 98) per il fit
    n_train, seq_len, n_feat = X_train.shape
    X_train_flat = X_train.reshape(-1, n_feat)
    scaler = StandardScaler()
    X_train_flat = scaler.fit_transform(X_train_flat)
    X_train = X_train_flat.reshape(n_train, seq_len, n_feat).astype(np.float32)

    n_val = X_val.shape[0]
    X_val_flat = X_val.reshape(-1, n_feat)
    X_val_flat = scaler.transform(X_val_flat)  # Usa lo STESSO scaler del train
    X_val = X_val_flat.reshape(n_val, seq_len, n_feat).astype(np.float32)

    print(f"\nX_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}    y_val:   {y_val.shape}")
    print(f"Distribuzione train → fight: {sum(y_train==1)}, no_fight: {sum(y_train==0)}")
    print(f"Distribuzione val   → fight: {sum(y_val==1)}, no_fight: {sum(y_val==0)}")

    # DataLoaders
    loader_train = DataLoader(FightDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True)
    loader_val = DataLoader(FightDataset(X_val, y_val),
                            batch_size=BATCH_SIZE, shuffle=False)

    # Modello
    modello = LSTMClassificatore().to(dispositivo)
    criterio_loss = nn.CrossEntropyLoss()
    ottimizzatore = optim.Adam(modello.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        ottimizzatore, mode='min', factor=0.5, patience=5
    )

    # Tracking
    storico_loss_train = []
    storico_loss_val = []
    storico_accuracy_val = []
    miglior_val_loss = float('inf')
    contatore_pazienza = 0

    os.makedirs(CARTELLA_MODELLI, exist_ok=True)
    percorso_modello = os.path.join(CARTELLA_MODELLI, "lstm_risse.pt")

    # Salva anche lo scaler (serve per inference!)
    import pickle
    with open(os.path.join(CARTELLA_MODELLI, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"Scaler salvato in: {CARTELLA_MODELLI}/scaler.pkl")

    print(f"\n{'='*60}")
    print(f" Inizio addestramento — max {EPOCHE} epoche")
    print(f"{'='*60}\n")

    for epoca in range(EPOCHE):
        # --- Training ---
        modello.train()
        perdita_train = 0.0
        for X_batch, y_batch in loader_train:
            X_batch, y_batch = X_batch.to(dispositivo), y_batch.to(dispositivo)
            predizioni = modello(X_batch)
            loss = criterio_loss(predizioni, y_batch)
            ottimizzatore.zero_grad()
            loss.backward()
            # Gradient clipping per stabilizzare LSTM
            torch.nn.utils.clip_grad_norm_(modello.parameters(), max_norm=1.0)
            ottimizzatore.step()
            perdita_train += loss.item()

        loss_train = perdita_train / len(loader_train)

        # --- Validazione ---
        modello.eval()
        perdita_val = 0.0
        corrette = 0
        totali = 0
        with torch.no_grad():
            for X_batch, y_batch in loader_val:
                X_batch, y_batch = X_batch.to(dispositivo), y_batch.to(dispositivo)
                predizioni = modello(X_batch)
                loss = criterio_loss(predizioni, y_batch)
                perdita_val += loss.item()
                corrette += (predizioni.argmax(1) == y_batch).sum().item()
                totali += y_batch.size(0)

        loss_val = perdita_val / len(loader_val)
        accuracy_val = corrette / totali

        # Aggiorna learning rate
        scheduler.step(loss_val)

        storico_loss_train.append(loss_train)
        storico_loss_val.append(loss_val)
        storico_accuracy_val.append(accuracy_val)

        lr_attuale = ottimizzatore.param_groups[0]['lr']
        print(f"Epoca [{epoca+1:3d}/{EPOCHE}]  "
              f"Train Loss: {loss_train:.4f}  "
              f"Val Loss: {loss_val:.4f}  "
              f"Val Acc: {accuracy_val:.4f}  "
              f"LR: {lr_attuale:.6f}")

        # Early stopping
        if loss_val < miglior_val_loss:
            miglior_val_loss = loss_val
            contatore_pazienza = 0
            torch.save(modello.state_dict(), percorso_modello)
            print(f"  ✓ Modello salvato (miglior val_loss: {miglior_val_loss:.4f})")
        else:
            contatore_pazienza += 1
            print(f"  ✗ Nessun miglioramento ({contatore_pazienza}/{PAZIENZA})")
            if contatore_pazienza >= PAZIENZA:
                print(f"\n⚠ Early stopping dopo {PAZIENZA} epoche senza miglioramento.")
                break

    print(f"\n{'='*60}")
    print(f" Addestramento completato — {epoca+1} epoche")
    print(f" Modello salvato in: {percorso_modello}")
    print(f"{'='*60}")

    # --- Metriche finali ---
    modello.load_state_dict(torch.load(percorso_modello, map_location=dispositivo))
    modello.eval()

    tutte_pred = []
    tutte_label = []
    with torch.no_grad():
        for X_batch, y_batch in loader_val:
            X_batch = X_batch.to(dispositivo)
            pred = modello(X_batch).argmax(1).cpu().numpy()
            tutte_pred.extend(pred)
            tutte_label.extend(y_batch.numpy())

    tutte_pred = np.array(tutte_pred)
    tutte_label = np.array(tutte_label)

    print(f"\n{'='*60}")
    print(" METRICHE DI VALUTAZIONE")
    print(f"{'='*60}\n")
    print(classification_report(tutte_label, tutte_pred,
                                target_names=["no_fight", "fight"]))

    cm = confusion_matrix(tutte_label, tutte_pred)
    print("Matrice di confusione:")
    print(cm)

    # --- Grafici ---
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
    plt.plot(epoche_range, storico_accuracy_val, label="Val Accuracy",
             marker='o', markersize=3, color='green')
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
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=20, fontweight='bold')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("grafici/confusion_matrix.png", dpi=150)
    plt.close()
    print("Grafico salvato: grafici/confusion_matrix.png")

    print(f"\n{'='*60}")
    print(" TUTTO COMPLETATO!")
    print(f" Modello:  {percorso_modello}")
    print(f" Scaler:   {CARTELLA_MODELLI}/scaler.pkl")
    print(f" Grafici:  grafici/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()