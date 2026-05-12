# Guida — Come costruire `2_addestra_lstm.py`

## Cosa deve fare questo script?

Lo script 2 ha **un unico compito**: prendere i CSV generati dallo script 1 e addestrare la rete LSTM. Alla fine produce un file `.pt` (il modello addestrato) che poi lo script 3 di inferenza userà.

Niente webcam, niente YOLO, niente video. Solo **dati numerici → rete neurale → modello salvato**.

---

## Da dove arrivano i dati?

Lo script 1 ha creato questa struttura:

```
output/
├── train/
│   ├── fight/
│   │   ├── video_001.csv
│   │   ├── video_002.csv
│   │   └── ...
│   └── no_fight/
│       ├── video_100.csv
│       └── ...
└── val/
    ├── fight/
    │   └── ...
    └── no_fight/
        └── ...
```

Ogni CSV ha questa forma (aprine uno per verificare):

```csv
frame, id_persona, naso_x, naso_y, occhio_sx_x, occhio_sx_y, ..., caviglia_dx_x, caviglia_dx_y
0,     1,          0.5123, 0.3456, 0.5200,      0.3300,      ..., 0.4800,        0.8900
0,     2,          0.7100, 0.4000, 0.7150,      0.3950,      ..., 0.6900,        0.9100
1,     1,          0.5130, 0.3460, 0.5210,      0.3310,      ..., 0.4810,        0.8910
1,     2,          0.7110, 0.4010, 0.7160,      0.3960,      ..., 0.6910,        0.9110
...
```

Ogni riga = **una persona in un frame**. Le coordinate sono già normalizzate [0,1] dallo script 1. 
Le prime 2 colonne sono `frame` e `id_persona`, le restanti 34 colonne sono le coordinate dei keypoints.

---

## Il problema chiave: da righe CSV a sequenze temporali

L'LSTM non mangia righe singole. Ha bisogno di **sequenze** — cioè una serie di frame consecutivi della stessa persona. Qui c'è il passaggio cruciale che devi capire bene.

### Passo 1 — Raggruppa per persona

Dentro un CSV (= un video), ci sono più persone tracciate con ID diversi. Prima di tutto devi **separare i dati per `id_persona`**:

```
CSV del video_001:
  Persona 1: frame 0, frame 1, frame 2, ..., frame 150  → 151 righe
  Persona 2: frame 0, frame 1, frame 3, ..., frame 140  → 130 righe (magari sparisce prima)
  Persona 3: frame 20, frame 21, ..., frame 80           → 61 righe (appare dopo)
```

Ogni persona diventa una **traiettoria indipendente** di 34 features nel tempo.

### Passo 2 — Sliding window (Cap. 3, §3.4)

La tesi dice:

> *"La dimensione della finestra è stata fissata a 30 fotogrammi consecutivi"*
> *"La finestra avanza con un passo (stride) inferiore a 30, garantendo una sovrapposizione"*

Quindi prendi la traiettoria di ogni persona e ci fai scorrere sopra una finestra di 30 frame:

```
Persona 1 ha 151 frame. Con stride = 15:

Sequenza 0:  frame [0  ... 29]   → shape (30, 34)
Sequenza 1:  frame [15 ... 44]   → shape (30, 34)
Sequenza 2:  frame [30 ... 59]   → shape (30, 34)
...
Sequenza 8:  frame [120 ... 149] → shape (30, 34)

→ Da una persona in un video ottieni ~9 sequenze di addestramento
```

Se una persona ha meno di 30 frame → **la scarti** (non ha abbastanza dati temporali).

### Passo 3 — Assegna la label

La label viene dalla **cartella** in cui si trova il CSV:
- CSV in `output/train/fight/` → label = **1** (rissa)
- CSV in `output/train/no_fight/` → label = **0** (non rissa)

Tutte le sequenze estratte da un video "fight" sono etichettate come fight.

### Risultato finale

Dopo aver processato tutti i CSV di tutti i video, hai:

```
X = array di sequenze, shape: (N_totale, 30, 34)
y = array di label,     shape: (N_totale,)

Dove N_totale è la somma di tutte le sequenze estratte da tutte le persone di tutti i video
```

Questo è il tensore `(batch_size, time_steps, features)` = `(N, 30, 34)` descritto nel Cap. 3.

---

## Struttura dello script — Le 6 sezioni

Lo script si divide logicamente in queste parti:

### Sezione 1 — Configurazione

Tutti i parametri in cima al file, facili da modificare per gli esperimenti:

```
- Percorsi delle cartelle (input CSV, output modello)
- Parametri sliding window (finestra=30, stride=15)
- Iperparametri LSTM (hidden size, dropout, ecc.)
- Parametri training (epoche, batch size, learning rate)
```

### Sezione 2 — Funzione per creare le sequenze da un CSV

Una funzione che:
1. Legge un CSV con `pandas` o il modulo `csv`
2. Raggruppa le righe per `id_persona`
3. Per ogni persona, applica la sliding window
4. Ritorna una lista di sequenze `(30, 34)`

Concettualmente:
```
input:  percorso_csv (es. "output/train/fight/video_001.csv")
output: lista di array numpy, ciascuno shape (30, 34)
```

### Sezione 3 — Caricamento di tutto il dataset

Un loop che:
1. Scorre `output/train/fight/*.csv` → crea sequenze con label 1
2. Scorre `output/train/no_fight/*.csv` → crea sequenze con label 0
3. Stessa cosa per `output/val/` (che sarà il tuo test set, il 20%)
4. Converte tutto in tensori PyTorch

> [!NOTE]
> **Nota sullo split**: Lo script 1 ha già separato `train/` e `val/`. Quindi non devi fare tu lo split 80/20 — è già fatto a livello di video. Il train set usa i CSV in `output/train/`, il validation/test set usa quelli in `output/val/`.

### Sezione 4 — Definizione del modello LSTM

Una classe PyTorch (`nn.Module`) che implementa **esattamente** l'architettura del Cap. 3, §3.4:

```
Blocco Estrazione Temporale:
  LSTM_1(input=34, hidden=128, return_sequences=True)
  Dropout
  LSTM_2(input=128, hidden=64, return_sequences=False)
  Dropout

Blocco Classificazione:
  Dense_1(64 → ?) + ReLU + Dropout      ← dimensione ridotta progressivamente
  Dense_2(? → ?) + ReLU + Dropout
  Dense_3(? → 2) + Softmax              ← 2 classi: fight, no_fight
```

> [!IMPORTANT]
> **Dettaglio PyTorch su LSTM**: In PyTorch non esiste il parametro `return_sequences`. L'`nn.LSTM` ritorna **sempre** l'intera sequenza. Per simulare `return_sequences=False` (2° layer), prendi semplicemente **l'ultimo timestep** dell'output: `output[:, -1, :]`.
>
> Per fare 2 layer separati (con hidden size diversi), hai due opzioni:
> - **Opzione A**: Due `nn.LSTM` separati (uno con hidden=128, uno con hidden=64)
> - **Opzione B**: Un unico `nn.LSTM(num_layers=2)` — ma in questo caso il hidden size è uguale per entrambi, il che **non** corrisponde a quello che vuoi
>
> Usa l'**Opzione A** (due LSTM separati) per avere 128 e 64 diversi.

> [!IMPORTANT]
> **Softmax + CrossEntropyLoss in PyTorch**: Attenzione — `nn.CrossEntropyLoss` in PyTorch **include già il Softmax internamente**. Quindi nel `forward()` del modello **NON** metti `nn.Softmax`. Restituisci i logit grezzi dall'ultimo Dense layer. Il Softmax è implicito nella loss. Se poi vuoi le probabilità (per l'inferenza), applichi `torch.softmax()` a mano sull'output.

### Sezione 5 — Training loop

Il classico ciclo di addestramento PyTorch:

```
Per ogni epoca:
    1. Modalità train → per ogni batch del train set:
       - Forward pass (dati → modello → predizioni)
       - Calcola loss
       - Backward pass (backpropagation)
       - Aggiorna pesi (optimizer.step)
    
    2. Modalità eval → per ogni batch del val set:
       - Solo forward pass (senza backprop)
       - Calcola loss e accuracy di validazione
    
    3. Early stopping:
       - Se la val_loss non migliora per N epoche → fermati
       - Salva il modello solo quando la val_loss migliora (best model)
    
    4. Stampa: epoca, train_loss, val_loss, val_accuracy
```

### Sezione 6 — Valutazione finale e metriche

Dopo il training, carica il modello migliore e valuta sul validation set:
- Accuracy, Precision, Recall, F1 (con `sklearn.metrics`)
- Matrice di confusione
- Grafici di loss e accuracy nel tempo (con `matplotlib`)
- Salva i grafici come immagini `.png` per metterli nella tesi

---

## Schema riassuntivo del flusso dati

```
output/train/fight/video_001.csv ──┐
output/train/fight/video_002.csv ──┤
...                                ├──→ Leggi CSV
output/train/no_fight/video_N.csv ─┘    Raggruppa per id_persona
                                        Sliding window (30, stride 15)
                                        Assegna label (0 o 1)
                                             │
                                             ▼
                                    X_train: (N, 30, 34)
                                    y_train: (N,)
                                             │
                                             ▼
                                      DataLoader PyTorch
                                     (batch=32, shuffle)
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │  Training LSTM   │
                                    │  (50 epoche max) │
                                    │  + early stopping│
                                    └────────┬────────┘
                                             │
                                             ▼
                                    modelli/lstm_risse.pt
                                    grafici/loss.png
                                    grafici/confusion_matrix.png
```

---

## Checklist — Cosa implementare

- [ ] Configurazione iperparametri in cima
- [ ] Funzione `crea_sequenze_da_csv(percorso_csv)` → lista di array (30, 34)
- [ ] Funzione `carica_dataset(cartella_output)` → X_train, y_train, X_val, y_val
- [ ] Classe `FightDataset(Dataset)` per PyTorch
- [ ] Classe `LSTMClassificatore(nn.Module)` — architettura del Cap. 3
- [ ] Training loop con early stopping
- [ ] Valutazione con metriche (accuracy, precision, recall, F1, confusion matrix)
- [ ] Grafici di loss/accuracy
- [ ] Salvataggio modello `.pt`
