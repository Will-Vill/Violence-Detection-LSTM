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

# FEATURES MULTI-PERSONA (per frame):
#   Persona A (la più vicina):  34 coordinate + 34 velocità = 68
#   Persona B (il suo vicino):  34 coordinate + 34 velocità = 68
#   Distanza tra i centri:      1
#   Variazione della distanza:  1  (negativa = si avvicinano, positiva = si allontanano)
#   Numero persone nel frame:   1
#   TOTALE:                     139
NUM_FEATURES = 139

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
# PREPARAZIONE DATI — VERSIONE MULTI-PERSONA
# ══════════════════════════════════════════════════════════════════════

def crea_sequenze_da_csv(percorso_csv):
    """
    Versione multi-persona: per ogni frame del video, individua la coppia
    di persone più vicine e costruisce un vettore di features congiunto.
    Le sequenze vengono create per VIDEO (non per persona), eliminando
    il rumore degli spettatori che affliggeva il modello base.

    Struttura del vettore per frame (139 features):
        [coord_A(34), coord_B(34), vel_A(34), vel_B(34),
         distanza(1), delta_distanza(1), num_persone(1)]
    """
    df = pd.read_csv(percorso_csv)

    if df.empty:
        return []

    # --- PASSO 1: Raccogli i keypoints di ogni persona per frame ---
    tutti_frame = sorted(df['frame'].unique())
    if len(tutti_frame) < 2:
        return []

    # Costruisci un range continuo di frame (riempi eventuali buchi)
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
            # Centro di massa = media di tutte le coordinate x e y
            x_coords = coords[0::2]
            y_coords = coords[1::2]
            centri[pid] = np.array([np.mean(x_coords), np.mean(y_coords)])
        dati_per_frame[frame_num] = (persone, centri)

    # --- PASSO 2: Per ogni frame, trova la coppia più vicina ---
    lista_coord_A = []
    lista_coord_B = []
    lista_distanze = []
    lista_num_persone = []

    for frame_num in frame_range:
        if frame_num not in dati_per_frame:
            # Frame senza nessuna persona rilevata (buco nel tracking)
            lista_coord_A.append(np.zeros(KP_SIZE))
            lista_coord_B.append(np.zeros(KP_SIZE))
            lista_distanze.append(1.0)
            lista_num_persone.append(0)
            continue

        persone, centri = dati_per_frame[frame_num]
        ids = list(persone.keys())
        n_persone = len(ids)

        if n_persone == 1:
            # Una sola persona: la seconda viene riempita con zeri
            lista_coord_A.append(persone[ids[0]])
            lista_coord_B.append(np.zeros(KP_SIZE))
            lista_distanze.append(1.0)
            lista_num_persone.append(1)

        else:
            # Due o più persone: trova la coppia con la distanza minima
            min_dist = float('inf')
            migliore_i = 0
            migliore_j = 1

            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    dist = np.sqrt(np.sum((centri[ids[i]] - centri[ids[j]]) ** 2))
                    if dist < min_dist:
                        min_dist = dist
                        migliore_i = i
                        migliore_j = j

            lista_coord_A.append(persone[ids[migliore_i]])
            lista_coord_B.append(persone[ids[migliore_j]])
            lista_distanze.append(min_dist)
            lista_num_persone.append(n_persone)

    # Converti in array numpy
    coord_A = np.array(lista_coord_A)       # (T, 34)
    coord_B = np.array(lista_coord_B)       # (T, 34)
    distanze = np.array(lista_distanze).reshape(-1, 1)      # (T, 1)
    num_persone = np.array(lista_num_persone).reshape(-1, 1) # (T, 1)

    if len(coord_A) < 2:
        return []

    # --- PASSO 3: Calcolo delle velocità ---
    # Per il primo frame la velocità è zero
    vel_A = np.diff(coord_A, axis=0)
    vel_A = np.vstack([np.zeros((1, KP_SIZE)), vel_A])

    vel_B = np.diff(coord_B, axis=0)
    vel_B = np.vstack([np.zeros((1, KP_SIZE)), vel_B])

    # --- PASSO 4: Variazione della distanza (delta) ---
    # Negativo = le persone si avvicinano (indicatore di collisione)
    # Positivo = le persone si allontanano
    delta_dist = np.diff(distanze, axis=0)
    delta_dist = np.vstack([np.zeros((1, 1)), delta_dist])

    # --- PASSO 5: Concatena tutto ---
    # coord_A(34) + coord_B(34) + vel_A(34) + vel_B(34) +
    # distanza(1) + delta_distanza(1) + num_persone(1) = 139
    features = np.hstack([
        coord_A, coord_B,
        vel_A, vel_B,
        distanze, delta_dist,
        num_persone
    ])

    # --- PASSO 6: Sliding window per VIDEO ---
    # A differenza del modello base che creava finestre per PERSONA,
    # qui la finestra scorre sui frame del video. Ogni video genera
    # un numero limitato di sequenze, tutte con etichetta corretta.
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

    dataset_train = FightDataset(X_train, y_train)
    dataset_val = FightDataset(X_val, y_val)

    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True)
    loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False)

    modello = LSTMClassificatore().to(dispositivo)

    criterio_loss = nn.CrossEntropyLoss()

    ottimizzatore = optim.Adam(modello.parameters(), lr=LEARNING_RATE)


    # TRAINING
    # Liste per salvare i valori di ogni epoca per i grafici alla fine
    storico_loss_train = []
    storico_loss_val = []
    storico_accuracy_val = []

    # Variabili per l'early stopping
    miglior_val_loss = float('inf')  # Parte da infinito
    contatore_pazienza = 0

    # Crea la cartella per salvare il modello
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
            # Sposta i dati sullo stesso dispositivo
            X_batch = X_batch.to(dispositivo)
            y_batch = y_batch.to(dispositivo)

            # I dati passano attraverso la rete
            predizioni = modello(X_batch)
            loss = criterio_loss(predizioni, y_batch)

            # Calcola i gradienti e aggiorna i pesi
            ottimizzatore.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modello.parameters(), max_norm=1.0)
            ottimizzatore.step()

            perdita_totale_train += loss.item()

        # Media della loss
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

                # Conta le predizioni per calcolare l'accuracy
                classi_predette = predizioni.argmax(1)
                corrette += (classi_predette == y_batch).sum().item()
                totali += y_batch.size(0)

        loss_media_val = perdita_totale_val / len(loader_val)
        accuracy_val = corrette / totali

        # Salva i valori per i grafici
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
    # Carica il modello migliore
    modello.load_state_dict(torch.load(percorso_modello, map_location=dispositivo))
    modello.eval()

    # Raccogli tutte le predizioni sul validation set
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

    # Stampa il report completo (accuracy, precision, recall, F1)
    print(f"\n{'='*60}")
    print(" METRICHE DI VALUTAZIONE")
    print(f"{'='*60}\n")
    print(classification_report(tutte_label, tutte_predizioni,
                                target_names=["no_fight", "fight"]))

    # Matrice di confusione
    cm = confusion_matrix(tutte_label, tutte_predizioni)
    print("Matrice di confusione:")
    print(cm)


    # GRAFICI
    os.makedirs("grafici", exist_ok=True)
    epoche_range = range(1, len(storico_loss_train) + 1)

    # Grafico 1: Loss train vs val
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

    # Grafico 2: Accuracy di validazione
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

    # Grafico 3: Matrice di confusione
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["no_fight", "fight"])
    ax.set_yticklabels(["no_fight", "fight"])
    ax.set_xlabel("Predetto")
    ax.set_ylabel("Reale")
    ax.set_title("Matrice di Confusione — Multi-Persona")
    # Scrivi i numeri dentro le celle
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
