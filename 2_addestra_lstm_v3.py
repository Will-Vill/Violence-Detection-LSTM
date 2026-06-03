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

dispositivo = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Viene utilizzato il dispositivo: {dispositivo}")

CARTELLA_INPUT = "output"
CARTELLA_MODELLI = "modello"

# Iperparametri
FINESTRA = 30
STRIDE = 15
NUM_FEATURES = 69
HIDDEN_1 = 128
HIDDEN_2 = 64
DENSE_1 = 32
DENSE_2 = 16
DROPOUT = 0.4        # Aumentato per contrastare overfitting
NUM_CLASSI = 2
LEARNING_RATE = 0.0005  # Ridotto per curve più stabili
WEIGHT_DECAY = 1e-4     # Regolarizzazione L2 sui pesi
BATCH_SIZE = 32
EPOCHE = 50
PAZIENZA = 10

# Indici per lo swap left/right nel flip orizzontale (COCO 17 keypoints)
# Ogni coppia = (indice_sinistro, indice_destro) nei 17 keypoints
COPPIE_SIMMETRICHE = [
    (1, 2),   # occhio sx ↔ dx
    (3, 4),   # orecchio sx ↔ dx
    (5, 6),   # spalla sx ↔ dx
    (7, 8),   # gomito sx ↔ dx
    (9, 10),  # polso sx ↔ dx
    (11, 12), # anca sx ↔ dx
    (13, 14), # ginocchio sx ↔ dx
    (15, 16), # caviglia sx ↔ dx
]


def flip_orizzontale(sequenza):
    """
    Data Augmentation: specchia orizzontalmente una sequenza.
    - Le coordinate x vengono invertite (x → 1-x)
    - I keypoints sinistri e destri vengono scambiati
    - Le velocità vengono ricalcolate coerentemente
    - La distanza dal vicino resta invariata

    Input:  sequenza di forma (30, 69)
    Output: sequenza flippata di forma (30, 69)
    """
    flipped = sequenza.copy()

    for t in range(len(flipped)):
        # --- Coordinate (indici 0-33): flip x, swap left/right ---
        coords = flipped[t, :34].copy()

        # Inverti tutte le x (indici pari: 0,2,4,...,32)
        coords[0::2] = 1.0 - coords[0::2]

        # Scambia i keypoint sinistri con i destri
        for kp_sx, kp_dx in COPPIE_SIMMETRICHE:
            # Ogni keypoint occupa 2 posizioni (x,y)
            idx_sx = kp_sx * 2
            idx_dx = kp_dx * 2
            coords[idx_sx], coords[idx_dx] = coords[idx_dx], coords[idx_sx]
            coords[idx_sx+1], coords[idx_dx+1] = coords[idx_dx+1], coords[idx_sx+1]

        flipped[t, :34] = coords

        # --- Velocità (indici 34-67): stessa logica ---
        vel = flipped[t, 34:68].copy()
        vel[0::2] = -vel[0::2]  # Inverti velocità x (direzione opposta)

        for kp_sx, kp_dx in COPPIE_SIMMETRICHE:
            idx_sx = kp_sx * 2
            idx_dx = kp_dx * 2
            vel[idx_sx], vel[idx_dx] = vel[idx_dx], vel[idx_sx]
            vel[idx_sx+1], vel[idx_dx+1] = vel[idx_dx+1], vel[idx_sx+1]

        flipped[t, 34:68] = vel

        # --- Distanza (indice 68): resta invariata ---

    return flipped


def crea_sequenze_da_csv(percorso_csv, categoria, filtrare):
    """
    Legge un CSV, calcola velocità e distanza, applica sliding window.
    Se filtrare=True E categoria='fight', rimuove il 25% delle persone
    meno attive (probabili spettatori).
    """
    df = pd.read_csv(percorso_csv)
    sequenze_video = []

    # Centro di massa per il calcolo distanze
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

    # Filtro spettatori: solo su fight, solo su training
    persone_da_tenere = set(df['id_persona'].unique())

    if filtrare and categoria == "fight":
        velocita_per_persona = {}
        for id_persona, dati_persona in df.groupby('id_persona'):
            coords = dati_persona.iloc[:, 2:].values.astype(float)
            if len(coords) < 2:
                velocita_per_persona[id_persona] = 0.0
                continue
            diff = np.diff(coords, axis=0)
            vel_media = np.mean(np.abs(diff))
            velocita_per_persona[id_persona] = vel_media

        if len(velocita_per_persona) > 3:
            # Filtro meno aggressivo: togli solo il 25% più fermo
            valori_vel = list(velocita_per_persona.values())
            soglia = np.percentile(valori_vel, 25)  # 25° percentile

            persone_da_tenere = set()
            for pid, vel in velocita_per_persona.items():
                if vel >= soglia:
                    persone_da_tenere.add(pid)

            # Tieni almeno 2 persone
            if len(persone_da_tenere) < 2:
                top2 = sorted(velocita_per_persona.keys(),
                              key=lambda p: velocita_per_persona[p], reverse=True)[:2]
                persone_da_tenere = set(top2)

    # Costruisci le sequenze per persona
    for id_persona, dati_persona in df.groupby('id_persona'):
        if id_persona not in persone_da_tenere:
            continue

        coordinate = dati_persona.iloc[:, 2:].values
        lista_frame = dati_persona['frame'].values

        if len(coordinate) < 2:
            continue

        velocita = np.diff(coordinate, axis=0)
        velocita = np.vstack([np.zeros((1, coordinate.shape[1])), velocita])

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
        features = np.hstack([coordinate, velocita, distanze])

        for i in range(0, len(features) - FINESTRA + 1, STRIDE):
            fetta_video = features[i : i + FINESTRA]
            sequenze_video.append(fetta_video)

    return sequenze_video


def carica_dataset(split, filtrare):
    X = []
    y = []

    categorie = ["fight", "no_fight"]

    for categoria in categorie:
        etichetta = 1 if categoria == "fight" else 0
        cartella = os.path.join(CARTELLA_INPUT, split, categoria)
        lista_file_csv = glob.glob(os.path.join(cartella, "*.csv"))

        for file_csv in lista_file_csv:
            sequenze = crea_sequenze_da_csv(file_csv, categoria, filtrare)
            for sequenza in sequenze:
                X.append(sequenza)
                y.append(etichetta)

        n_cat = sum(1 for label in y if label == etichetta)
        print(f"  {categoria}: {n_cat} sequenze totali")

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
        x = x[:, -1, :]
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout4(x)
        x = self.fc3(x)
        return x


def main():
    # CARICAMENTO DATI
    # Training: CON filtro spettatori (rimuove il 25% meno attivo nei fight)
    print("Caricamento Train Set (con filtro spettatori)...")
    X_train, y_train = carica_dataset("train", filtrare=True)

    # Validation: SENZA filtro (testa su TUTTI, come il modello base)
    print("\nCaricamento Validation Set (SENZA filtro — test completo)...")
    X_val, y_val = carica_dataset("val", filtrare=False)

    if len(X_train) == 0 or len(X_val) == 0:
        print("ERRORE: Non ci sono abbastanza dati")
        return

    # --- DATA AUGMENTATION: flip orizzontale ---
    print("\nApplicazione Data Augmentation (flip orizzontale)...")
    X_aug = []
    y_aug = []
    for i in range(len(X_train)):
        X_aug.append(flip_orizzontale(X_train[i]))
        y_aug.append(y_train[i])

    X_train = np.concatenate([X_train, np.array(X_aug)], axis=0)
    y_train = np.concatenate([y_train, np.array(y_aug)], axis=0)

    # Shuffle dopo l'augmentation
    indici = np.random.permutation(len(X_train))
    X_train = X_train[indici]
    y_train = y_train[indici]

    print(f"  Sequenze dopo augmentation: {len(X_train)}")

    # --- NORMALIZZAZIONE ---
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

    # Adam con weight decay per regolarizzazione L2
    ottimizzatore = optim.Adam(modello.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

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
