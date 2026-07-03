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
CARTELLA_MODELLI = "modello"

# Costanti e iperparametri della rete
FINESTRA = 30
STRIDE = 15
NUM_FEATURES = 69  # 34 coordinate + 34 velocità + 1 distanza
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

# PREPARAZIONE DATI
def crea_sequenze_da_csv(percorso_csv):
    """Legge un CSV di keypoints, calcola velocità e distanza, e applica sliding window."""
    df = pd.read_csv(percorso_csv)
    
    sequenze_video = []

    # Pre-calcolo del centro di massa di ogni persona per frame
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

    # Per ogni persona, costruisci le sequenze
    for id_persona, dati_persona in df.groupby('id_persona'):
        coordinate = dati_persona.iloc[:, 2:].values
        lista_frame = dati_persona['frame'].values

        if len(coordinate) < 2:
            continue

        # Velocità: differenza tra frame consecutivi
        velocita = np.diff(coordinate, axis=0)
        velocita = np.vstack([np.zeros((1, coordinate.shape[1])), velocita])

        # Distanza dal vicino più prossimo
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

        # Concatena tutto: coordinate(34) + velocità(34) + distanza(1) = 69
        features = np.hstack([coordinate, velocita, distanze])

        for i in range(0, len(features) - FINESTRA + 1, STRIDE):
            fetta_video = features[i : i + FINESTRA]
            sequenze_video.append(fetta_video)

    return sequenze_video



def carica_dataset(split):
    """Costruisce i dataset finali associando etichette fight/no_fight."""

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
    """Dataset PyTorch per le sequenze di features."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]




class LSTMClassificatore(nn.Module):
    """Rete LSTM a 2 livelli con 3 livelli Fully Connected."""
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
        
        # Ultimo fotogramma della sequenza
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
    percorso_modello = os.path.join(CARTELLA_MODELLI, "modello_base.pt")

    print(f"\n{'='*60}")
    print(f" Inizio addestramento — max {EPOCHE} epoche")
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
    plt.title("Andamento della Loss durante il Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("grafici/loss.png", dpi=150)
    plt.close()
    print("\nGrafico salvato: grafici/loss.png")

    # Grafico 2: Accuracy di validazione
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

    # Grafico 3: Matrice di confusione (Assoluta)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    
    dim_labels = 14
    dim_titoli = 16
    fig_size = (6, 5)

    
    fig, ax = plt.subplots(figsize=fig_size)
    im = ax.imshow(cm, cmap='Blues')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["no_fight", "fight"], fontsize=dim_labels)
    ax.set_yticklabels(["no_fight", "fight"], fontsize=dim_labels)
    ax.set_xlabel("Predetto", fontsize=dim_titoli)
    ax.set_ylabel("Reale", fontsize=dim_titoli)

    thresh_assoluta = 2000

    for i in range(2):
        for j in range(2):
            colore_testo = "white" if cm[i, j] > thresh_assoluta else "black"
            ax.text(j, i, str(cm[i, j]), 
                    ha='center', va='center', 
                    fontsize=20, fontweight='bold', 
                    color=colore_testo)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("grafici/confusion_matrix.png", dpi=150)
    plt.close()
    print("Grafico 1 salvato: grafici/confusion_matrix.png")

    
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=fig_size)
    im = ax.imshow(cm_norm, cmap='Blues')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    
    
    ax.set_xticklabels(["no_fight", "fight"], fontsize=dim_labels)
    ax.set_xlabel("Predetto", fontsize=dim_titoli)

    
    ax.set_yticklabels(["no_fight", "fight"], fontsize=dim_labels, color='white')
    ax.set_ylabel("Reale", fontsize=dim_titoli, color='white')
    ax.tick_params(axis='y', colors='white')

    thresh_norm = 0.5

    for i in range(2):
        for j in range(2):
            colore_testo = "white" if cm_norm[i, j] > thresh_norm else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", 
                    ha='center', va='center', 
                    fontsize=20, fontweight='bold', 
                    color=colore_testo)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("grafici/confusion_matrix_norm.png", dpi=150)
    plt.close()
    print("Grafico 2 salvato: grafici/confusion_matrix_norm.png")
    print(f"\n{'='*60}")
    print(" TUTTO COMPLETATO!")
    print(f" Modello:  {percorso_modello}")
    print(f" Grafici:  grafici/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()