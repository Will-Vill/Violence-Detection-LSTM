# Analisi Architetturale — Integrazione YOLO Object Detection

## Il Problema

La pipeline principale della tesi è chiara e solida:

```
Video → YOLO-Pose → Keypoints (34 features) → Sequenze temporali → LSTM → Rissa / Non Rissa
```

La domanda è: **dove e come integrare YOLO Object Detection (rilevamento armi)?**

Ci sono due approcci possibili, analizzati di seguito.

---

## Opzione A — Integrare le armi nel vettore LSTM (nel training)

In questo approccio, le informazioni di YOLO Object Detection verrebbero **fuse nel vettore di input dell'LSTM** durante il training:

```
Per ogni frame, il vettore diventerebbe:
  34 features (keypoints) + K features (armi) = 34+K features

Esempio: aggiungere un flag binario "arma_presente" → 35 features
Esempio: aggiungere tipo_arma + distanza_da_persona → 36-38 features
```

### ❌ Problemi di questo approccio

| Problema | Spiegazione |
|---|---|
| **Dataset** | Servono video etichettati **sia** per rissa/non-rissa **sia** per presenza armi. I dataset RWF-2000 e Real Life Violence **non hanno annotazioni sulle armi**. Dovresti annotare tutto manualmente o trovare dataset specifici. |
| **YOLO standard non è affidabile sulle armi** | Le classi COCO (coltello=43, mazza da baseball=34) sono troppo generiche. Un coltello da cucina in un video di cucina verrebbe rilevato. Servirebbero dataset di fine-tuning specifici per armi in contesto violento. |
| **Complessità del training** | Mescolare features eterogenee (coordinate spaziali continue + flag binari categorici) può confondere l'LSTM e richiede una normalizzazione molto accurata. |
| **Accoppiamento eccessivo** | Se YOLO Object Detection sbaglia (falso positivo/negativo sulle armi), l'errore si propaga direttamente nella classificazione LSTM, degradando le performance sull'obiettivo principale (rissa/non-rissa). |

---

## Opzione B — YOLO Object Detection solo in inferenza, post-LSTM ✅ (Raccomandato)

In questo approccio — che è **esattamente la tua intuizione** — YOLO Object Detection interviene **solo nella fase di inferenza**, **dopo** che l'LSTM ha già classificato la scena:

```
Video → YOLO-Pose → LSTM → Classificazione base (Rissa / Non Rissa)
  │
  └──→ YOLO Object Detection → Arma presente? (Sì/No)
  
  Combinazione finale:
  ┌─────────────────────┬────────────────┬─────────────────────────────┐
  │ LSTM dice...        │ YOLO dice...   │ Output finale               │
  ├─────────────────────┼────────────────┼─────────────────────────────┤
  │ Rissa               │ Arma rilevata  │ 🔴 RISSA CON ARMI (critica)│
  │ Rissa               │ No arma        │ 🟠 RISSA (senza armi)      │
  │ Non rissa           │ Arma rilevata  │ 🟡 ATTENZIONE (arma vista) │
  │ Non rissa           │ No arma        │ 🟢 SITUAZIONE NORMALE      │
  └─────────────────────┴────────────────┴─────────────────────────────┘
```

### ✅ Vantaggi di questo approccio

| Vantaggio | Spiegazione |
|---|---|
| **Disaccoppiamento** | L'LSTM viene addestrato **solo** sui keypoints, che è il cuore scientifico della tesi. YOLO Object Detection è un modulo **indipendente** che non inquina il training. |
| **Nessun dataset aggiuntivo** | Non serve annotare armi nei video di training. YOLO Object Detection lavora con le classi COCO pre-addestrate. |
| **Robustezza** | Se YOLO sbaglia un rilevamento armi, la classificazione base (rissa/non-rissa) rimane intatta. L'informazione sulle armi è solo un **segnale di rinforzo**. |
| **Modularità** | Puoi aggiungere/rimuovere il modulo armi senza riaddestare nulla. Perfetto per una tesi: presenti risultati base e poi mostri l'estensione. |
| **Coerente con la tesi** | Nel capitolo 3 hai già scritto: *"Questo modulo opera esclusivamente durante la fase di inferenza, a valle della classificazione LSTM"* — l'Opzione B è esattamente questo. |

> [!IMPORTANT]
> **L'Opzione B è quella corretta e accademicamente più solida.** La tua intuizione era giusta. Mantieni la pipeline principale pulita (YOLO-Pose + LSTM) e aggiungi YOLO Object Detection come modulo di arricchimento semantico solo in inferenza.

---

## Architettura Aggiornata — Schema di Inferenza

```
                    ┌──────────────────┐
                    │   Video Input    │
                    │  (file / webcam) │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Frame Corrente  │
                    └───┬──────────┬───┘
                        │          │
           ┌────────────▼──┐  ┌───▼────────────┐
           │  YOLO26n-Pose │  │   YOLO26n      │
           │  (keypoints)  │  │ (object det.)  │
           └────────┬──────┘  └───┬────────────┘
                    │             │
           ┌────────▼──────┐     │
           │ Pre-elab. +   │     │
           │ Sliding Window│     │
           └────────┬──────┘     │
                    │             │
           ┌────────▼──────┐     │
           │     LSTM      │     │
           │ (classificaz.)│     │
           └────────┬──────┘     │
                    │             │
           ┌────────▼─────────────▼───┐
           │   Logica di Decisione    │
           │  (combinazione risultati)│
           └────────┬─────────────────┘
                    │
        ┌───────────┼───────────┬──────────────┐
        ▼           ▼           ▼              ▼
   🔴 Rissa    🟠 Rissa    🟡 Attenzione  🟢 Normale
   con armi    senza armi  (arma vista)
```

> [!NOTE]
> I due YOLO girano **in parallelo** sullo stesso frame. YOLO-Pose alimenta la pipeline LSTM. YOLO Object Detection fornisce il suo risultato alla logica di decisione finale. Sono **indipendenti** e non si influenzano a vicenda.

---

## Impatto sugli Script

### Script 1 — `1_estrai_keypoints.py` (già fatto ✅)
Nessuna modifica. Continua a estrarre solo i keypoints con YOLO-Pose. YOLO Object Detection **non serve** nella fase di training.

### Script 2 — `2_addestra_lstm.py` (da fare)
Nessuna modifica architetturale. L'LSTM si addestra solo sui vettori a 34 features dei keypoints. L'input shape rimane `(batch, 30, 34)`.

### Script 3 — Inferenza (da fare)
Qui serve la decisione più importante:

#### Un solo script o due?

> [!TIP]
> **Un solo script con argomento `--source`.** Non ha senso dividere in due script, è lo stesso flusso logico con una sorgente diversa.

```python
# Esempio di utilizzo:
python 3_inferenza.py --source video.mp4           # da file
python 3_inferenza.py --source 0                   # da webcam
python 3_inferenza.py --source video.mp4 --no-armi # senza modulo armi
```

La logica interna sarebbe:

```python
# Pseudo-codice dello script di inferenza
modello_pose = YOLO("yolo26n-pose.pt")
modello_lstm = carica_modello_lstm("lstm_risse.pt")
modello_armi = YOLO("yolo26n.pt")  # opzionale

buffer = CircularBuffer(size=30)

for frame in video_source:
    # Ramo 1: YOLO-Pose → keypoints
    keypoints = modello_pose(frame)
    buffer.append(keypoints)
    
    if buffer.is_full():
        # LSTM classifica
        prob_rissa = modello_lstm(buffer.to_tensor())
        is_rissa = prob_rissa > SOGLIA
        
        # Ramo 2: YOLO Object Detection → armi
        armi_rilevate = modello_armi(frame)  # classi: coltello, mazza, ecc.
        has_arma = len(armi_rilevate) > 0
        
        # Logica di decisione finale
        if is_rissa and has_arma:
            output = "🔴 RISSA CON ARMI"
        elif is_rissa:
            output = "🟠 RISSA"
        elif has_arma:
            output = "🟡 ATTENZIONE - ARMA RILEVATA"
        else:
            output = "🟢 NORMALE"
```

---

## Impatto sul Diagramma TikZ

Il tuo diagramma attuale è **già quasi corretto** per l'Opzione B. L'unica modifica suggerita sarebbe:

1. **Rendere il blocco YOLO26n tratteggiato** (come già definito nello style `block dashed`) per indicare che è opzionale/secondario
2. **La freccia da YOLO26n NON va al blocco Pre-elaborazione**, ma va direttamente a un nuovo blocco **"Logica di Decisione"** che sta tra LSTM e Output
3. Aggiornare l'output con 3-4 livelli (rissa con armi, rissa senza armi, attenzione arma, normale)

> [!NOTE]
> Se vuoi, posso aggiornare il file `architettura_tikz.tex` per riflettere questa architettura. Fammi sapere.

---

## Riepilogo della Raccomandazione

| Aspetto | Decisione |
|---|---|
| Pipeline principale | **YOLO-Pose → Pre-elaborazione → LSTM** (invariata) |
| YOLO Object Detection | **Solo in inferenza**, post-LSTM, come segnale di rinforzo |
| Integrazione nel vettore LSTM | **No** — mantieni 34 features pure |
| Script di inferenza | **Uno solo** con `--source` per file/webcam |
| Output classificazione | **4 livelli**: rissa+armi, rissa, attenzione arma, normale |

> [!IMPORTANT]
> Questa architettura è coerente con quello che hai già scritto nel Capitolo 3 della tesi (sezione "Panoramica dell'architettura"). Non devi riscrivere nulla nel testo, devi solo aggiornare il diagramma TikZ per riflettere meglio il flusso post-LSTM.
