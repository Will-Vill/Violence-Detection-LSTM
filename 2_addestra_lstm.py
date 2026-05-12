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
# SETUP DEVICE (Hardware)
# ==========================================
# Questa riga cerca CUDA (NVIDIA), poi MPS (Mac Apple Silicon), altrimenti usa la CPU
dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Viene utilizzato il dispositivo: {dispositivo}")
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


def carica_dataset(split):
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
    # ==========================================
    # BLOCCO 1: Caricamento dati e setup
    # ==========================================
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

    # ==========================================
    # BLOCCO 2 + 3: Training loop con early stopping
    # ==========================================
    # Liste per salvare i valori di ogni epoca (servono per i grafici alla fine)
    storico_loss_train = []
    storico_loss_val = []
    storico_accuracy_val = []

    # Variabili per l'early stopping
    miglior_val_loss = float('inf')  # Parte da infinito, così qualsiasi loss è migliore
    contatore_pazienza = 0

    # Crea la cartella per salvare il modello
    os.makedirs(CARTELLA_MODELLI, exist_ok=True)
    percorso_modello = os.path.join(CARTELLA_MODELLI, "lstm_risse.pt")

    print(f"\n{'='*60}")
    print(f" Inizio addestramento — max {EPOCHE} epoche")
    print(f"{'='*60}\n")

    for epoca in range(EPOCHE):

        # --- FASE TRAIN ---
        modello.train()  # Attiva dropout e batch norm (modalità addestramento)
        perdita_totale_train = 0.0

        for X_batch, y_batch in loader_train:
            # Sposta i dati sullo stesso dispositivo del modello (GPU/MPS/CPU)
            X_batch = X_batch.to(dispositivo)
            y_batch = y_batch.to(dispositivo)

            # Forward pass: i dati passano attraverso la rete
            predizioni = modello(X_batch)
            loss = criterio_loss(predizioni, y_batch)

            # Backward pass: calcola i gradienti e aggiorna i pesi
            ottimizzatore.zero_grad()  # Azzera i gradienti dell'iterazione precedente
            loss.backward()            # Calcola i nuovi gradienti (backpropagation)
            ottimizzatore.step()       # Aggiorna i pesi della rete

            perdita_totale_train += loss.item()

        # Media della loss su tutti i batch del train
        loss_media_train = perdita_totale_train / len(loader_train)

        # --- FASE VALIDAZIONE ---
        modello.eval()  # Disattiva dropout (modalità valutazione)
        perdita_totale_val = 0.0
        corrette = 0
        totali = 0

        with torch.no_grad():  # Non calcolare gradienti (risparmia memoria)
            for X_batch, y_batch in loader_val:
                X_batch = X_batch.to(dispositivo)
                y_batch = y_batch.to(dispositivo)

                predizioni = modello(X_batch)
                loss = criterio_loss(predizioni, y_batch)
                perdita_totale_val += loss.item()

                # Conta le predizioni corrette per calcolare l'accuracy
                # .argmax(1) prende l'indice della classe con il valore più alto
                classi_predette = predizioni.argmax(1)
                corrette += (classi_predette == y_batch).sum().item()
                totali += y_batch.size(0)

        loss_media_val = perdita_totale_val / len(loader_val)
        accuracy_val = corrette / totali

        # Salva i valori per i grafici
        storico_loss_train.append(loss_media_train)
        storico_loss_val.append(loss_media_val)
        storico_accuracy_val.append(accuracy_val)

        # Stampa il progresso
        print(f"Epoca [{epoca+1:3d}/{EPOCHE}]  "
              f"Train Loss: {loss_media_train:.4f}  "
              f"Val Loss: {loss_media_val:.4f}  "
              f"Val Accuracy: {accuracy_val:.4f}")

        # --- EARLY STOPPING ---
        if loss_media_val < miglior_val_loss:
            # La val_loss è migliorata → salva il modello e resetta il contatore
            miglior_val_loss = loss_media_val
            contatore_pazienza = 0
            torch.save(modello.state_dict(), percorso_modello)
            print(f"  ✓ Modello salvato (miglior val_loss: {miglior_val_loss:.4f})")
        else:
            # La val_loss NON è migliorata → incrementa il contatore
            contatore_pazienza += 1
            print(f"  ✗ Nessun miglioramento ({contatore_pazienza}/{PAZIENZA})")
            if contatore_pazienza >= PAZIENZA:
                print(f"\n⚠ Early stopping! La val_loss non migliora da {PAZIENZA} epoche.")
                break

    print(f"\n{'='*60}")
    print(f" Addestramento completato — {epoca+1} epoche")
    print(f" Modello salvato in: {percorso_modello}")
    print(f"{'='*60}")

    # ==========================================
    # BLOCCO 4: Valutazione finale e metriche
    # ==========================================
    # Carica il modello migliore (quello salvato durante il training)
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

    # ==========================================
    # GRAFICI
    # ==========================================
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

    # Grafico 3: Matrice di confusione
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["no_fight", "fight"])
    ax.set_yticklabels(["no_fight", "fight"])
    ax.set_xlabel("Predetto")
    ax.set_ylabel("Reale")
    ax.set_title("Matrice di Confusione")
    # Scrivi i numeri dentro le celle
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