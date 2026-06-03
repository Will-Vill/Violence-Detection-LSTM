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
from collections import defaultdict
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
CARTELLA_MODELLI = "modelli"

# Costanti e iperparametri della rete
FINESTRA = 30
STRIDE = 15

# FEATURES MULTI-PERSONA RIDOTTE (per frame):
#   Persona A (della coppia):   34 coordinate + 34 velocità = 68
#   Distanza tra A e B:         1
#   Variazione della distanza:  1  (negativa = si avvicinano)
#   Numero persone nel frame:   1
#   TOTALE:                     71
#
# NOTA: le coordinate raw di persona B sono state RIMOSSE perché
# causavano overfitting — la rete le usava per memorizzare il video
# di provenienza, non per capire se c'era una rissa.
NUM_FEATURES = 71

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

NUM_KP = 17
KP_SIZE = NUM_KP * 2  # 34 valori (17 keypoints x 2 coordinate)


# ══════════════════════════════════════════════════════════════════════
# PREPARAZIONE DATI — VERSIONE MULTI-PERSONA CON FEATURES RIDOTTE
# ══════════════════════════════════════════════════════════════════════

def crea_sequenze_da_csv(percorso_csv):
    """
    Versione multi-persona con features ridotte e tracking consistente.

    Come nella versione multi base:
      - Identifica la coppia dominante del video e la segue.
      - Le sequenze vengono create per VIDEO (sliding window).

    Features ridotte (71 invece di 139):
      - Si mantengono solo le coordinate e velocità di PERSONA A.
      - Di persona B si tiene solo la RELAZIONE (distanza, delta, num_persone).
      - Le coordinate raw di B sono state rimosse perché causavano
        overfitting senza aggiungere capacità discriminativa.

    Struttura del vettore per frame (71 features):
        [coord_A(34), vel_A(34), distanza(1), delta_distanza(1), num_persone(1)]
    """
    df = pd.read_csv(percorso_csv)

    if df.empty:
        return []

    # --- PASSO 1: Raccogli i keypoints di ogni persona per frame ---
    tutti_frame = sorted(df['frame'].unique())
    if len(tutti_frame) < 2:
        return []

    frame_min = tutti_frame[0]
    frame_max = tutti_frame[-1]
    frame_range = list(range(frame_min, frame_max + 1))

    # Per ogni frame, salva le coordinate di ogni persona e il suo centro
    dati_per_frame = {}
    for frame_num, dati_frame in df.groupby('frame'):
        persone = {}
        centri = {}
        for _, riga in dati_frame.iterrows():
            pid = int(riga['id_persona'])
            coords = riga.iloc[2:].values.astype(float)  # 34 valori
            persone[pid] = coords
            x_coords = coords[0::2]
            y_coords = coords[1::2]
            centri[pid] = np.array([np.mean(x_coords), np.mean(y_coords)])
        dati_per_frame[frame_num] = (persone, centri)

    # --- PASSO 2: Identifica la coppia dominante del video ---
    conteggio_coppie = defaultdict(int)
    somma_distanze = defaultdict(float)

    for frame_num in frame_range:
        if frame_num not in dati_per_frame:
            continue
        persone, centri = dati_per_frame[frame_num]
        ids = sorted(persone.keys())

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                coppia = (ids[i], ids[j])
                dist = np.sqrt(np.sum((centri[ids[i]] - centri[ids[j]]) ** 2))
                conteggio_coppie[coppia] += 1
                somma_distanze[coppia] += dist

    if len(conteggio_coppie) == 0:
        freq = defaultdict(int)
        for frame_num in frame_range:
            if frame_num in dati_per_frame:
                persone, _ = dati_per_frame[frame_num]
                for pid in persone:
                    freq[pid] += 1
        if not freq:
            return []
        id_A = max(freq, key=freq.get)
        id_B = -999
    else:
        migliore = max(
            conteggio_coppie.keys(),
            key=lambda c: (conteggio_coppie[c],
                           -somma_distanze[c] / conteggio_coppie[c])
        )
        id_A, id_B = migliore

    # --- PASSO 3: Estrai le coordinate di A e la distanza da B ---
    lista_coord_A = []
    lista_distanze = []
    lista_num_persone = []

    ultimo_A = np.zeros(KP_SIZE)

    for frame_num in frame_range:
        if frame_num not in dati_per_frame:
            lista_coord_A.append(ultimo_A.copy())
            lista_distanze.append(lista_distanze[-1] if lista_distanze else 1.0)
            lista_num_persone.append(0)
            continue

        persone, centri = dati_per_frame[frame_num]
        n_persone = len(persone)

        # Estrai persona A (forward-fill se assente)
        if id_A in persone:
            coord_a = persone[id_A]
            ultimo_A = coord_a.copy()
        else:
            coord_a = ultimo_A.copy()

        lista_coord_A.append(coord_a)

        # Calcola solo la DISTANZA da B (non le sue coordinate)
        if id_B in centri and id_A in centri:
            dist = np.sqrt(np.sum((centri[id_A] - centri[id_B]) ** 2))
        elif n_persone >= 2 and id_A in centri:
            # B non presente ma ci sono altre persone: usa la più vicina
            min_d = float('inf')
            for pid in centri:
                if pid != id_A:
                    d = np.sqrt(np.sum((centri[id_A] - centri[pid]) ** 2))
                    if d < min_d:
                        min_d = d
            dist = min_d
        else:
            dist = 1.0

        lista_distanze.append(dist)
        lista_num_persone.append(n_persone)

    # Converti in array numpy
    coord_A = np.array(lista_coord_A)       # (T, 34)
    distanze = np.array(lista_distanze).reshape(-1, 1)      # (T, 1)
    num_persone = np.array(lista_num_persone).reshape(-1, 1) # (T, 1)

    if len(coord_A) < 2:
        return []

    # --- PASSO 4: Calcolo delle velocità (solo persona A) ---
    vel_A = np.diff(coord_A, axis=0)
    vel_A = np.vstack([np.zeros((1, KP_SIZE)), vel_A])

    # --- PASSO 5: Variazione della distanza ---
    delta_dist = np.diff(distanze, axis=0)
    delta_dist = np.vstack([np.zeros((1, 1)), delta_dist])

    # --- PASSO 6: Concatena tutto ---
    # coord_A(34) + vel_A(34) + distanza(1) + delta_distanza(1) + num_persone(1) = 71
    features = np.hstack([
        coord_A,
        vel_A,
        distanze, delta_dist,
        num_persone
    ])

    # --- PASSO 7: Sliding window per VIDEO ---
    sequenze_video = []
    for i in range(0, len(features) - FINESTRA + 1, STRIDE):
        fetta_video = features[i : i + FINESTRA]
        sequenze_video.append(fetta_video)

    return sequenze_video



def carica_dataset(split):
    """
    Cerca le cartelle (train/val) e costruisce i dataset finali,
    associa l'etichetta 1 per fight e 0 per no_fight e ritorna gli array
    numpy per essere convertiti in tensori.
    """

    X = []
    y = []

    categorie = ["fight", "no_fight"]

    for categoria in categorie:
        if categoria == "fight":
            etichetta = 1
        else:
            etichetta = 0
        cartella = os.path.join(CARTELLA_INPUT, split, categoria)
        lista_file_csv = glob.glob(os.path.join(cartella, "*.csv"))

        for file_csv in lista_file_csv:
            sequenze = crea_sequenze_da_csv(file_csv)

            print(f"  {categoria}: {len(sequenze)} sequenze da {file_csv}")

            for sequenza in sequenze:
                X.append(sequenza)
                y.append(etichetta)

    return np.array(X), np.array(y)



# CLASSI PYTORCH
class FightDataset(torch.utils.data.Dataset):
    """
    Converte gli array numpy di features (X) ed etichette (y) in tensori
    cosi da poter essere gestiti durante il training.
    """
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]




class LSTMClassificatore(nn.Module):
    """
    Architettura della rete neurale (Riferimento: Capitolo 3 della tesi).
    Composto da 2 livelli LSTM sequenziali e 3 livelli Fully Connected.
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
        # Passaggio nei blocchi LSTM
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        
        # Prendiamo solo l'ultimo fotogramma della sequenza
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
    # CARICAMENTO DATI
    print("Caricamento Train Set...")
    X_train, y_train = carica_dataset("train")

    print("\nCaricamento Validation Set...")
    X_val, y_val = carica_dataset("val")

    if len(X_train) == 0 or len(X_val) == 0:
        print("ERRORE: Non ci sono abbastanza dati per addestrare il modello")
        return

    print(f"\nX_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}    y_val:   {y_val.shape}")
    print(f"Distribuzione train → fight: {sum(y_train==1)}, no_fight: {sum(y_train==0)}")
    print(f"Distribuzione val   → fight: {sum(y_val==1)}, no_fight: {sum(y_val==0)}")

    # ── NORMALIZZAZIONE FEATURES ──
    scaler = StandardScaler()
    n_train, seq_len, n_feat = X_train.shape
    scaler.fit(X_train.reshape(-1, n_feat))

    X_train = scaler.transform(X_train.reshape(-1, n_feat)).reshape(n_train, seq_len, n_feat)
    n_val = X_val.shape[0]
    X_val = scaler.transform(X_val.reshape(-1, n_feat)).reshape(n_val, seq_len, n_feat)
    print("Features normalizzate con StandardScaler (fit su train)")

    dataset_train = FightDataset(X_train, y_train)
    dataset_val = FightDataset(X_val, y_val)

    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)

    modello = LSTMClassificatore().to(dispositivo)

    criterio_loss = nn.CrossEntropyLoss()

    ottimizzatore = optim.Adam(modello.parameters(), lr=LEARNING_RATE)


    # TRAINING
    storico_loss_train = []
    storico_loss_val = []
    storico_accuracy_val = []

    miglior_val_loss = float('inf')
    contatore_pazienza = 0

    os.makedirs(CARTELLA_MODELLI, exist_ok=True)
    percorso_modello = os.path.join(CARTELLA_MODELLI, "lstm_risse_multiv5.pt")

    print(f"\n{'='*60}")
    print(f" Inizio addestramento MULTI RIDOTTO — max {EPOCHE} epoche")
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

    
    # VALUTAZIONI FINALI E METRICHE
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
    plt.title("Andamento della Loss — Modello Multi Ridotto")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("grafici/loss_multiv5.png", dpi=150)
    plt.close()
    print("\nGrafico salvato: grafici/loss_multiv5.png")

    plt.figure(figsize=(10, 5))
    plt.plot(epoche_range, storico_accuracy_val, label="Val Accuracy", marker='o',
             markersize=3, color='green')
    plt.xlabel("Epoca")
    plt.ylabel("Accuracy")
    plt.title("Andamento dell'Accuracy — Modello Multi Ridotto")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("grafici/accuracy_multiv5.png", dpi=150)
    plt.close()
    print("Grafico salvato: grafici/accuracy_multiv5.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["no_fight", "fight"])
    ax.set_yticklabels(["no_fight", "fight"])
    ax.set_xlabel("Predetto")
    ax.set_ylabel("Reale")
    ax.set_title("Matrice di Confusione — Multi Ridotto")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=20, fontweight='bold')
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("grafici/confusion_matrix_multiv5.png", dpi=150)
    plt.close()
    print("Grafico salvato: grafici/confusion_matrix_multiv5.png")

    print(f"\n{'='*60}")
    print(" TUTTO COMPLETATO!")
    print(f" Modello:  {percorso_modello}")
    print(f" Grafici:  grafici/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
