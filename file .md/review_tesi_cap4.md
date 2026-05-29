# Review Tesi + Struttura Capitolo 4

---

## 1. Frontespizio

Il frontespizio è **ben fatto**. Nessun problema:
- Logo, dipartimento, corso di laurea ✅
- Nome studente in posizione corretta ✅
- Titolo chiaro e tra due filetti ✅
- Relatrice e Correlatore a destra ✅
- Anno Accademico in fondo ✅

Non c'è nulla da cambiare.

---

## 2. Capitolo 2 — Stato dell'Arte

### Qualità del contenuto: ⭐⭐⭐⭐ Buona

Il capitolo è solido, ben strutturato e copre quello che serve. La logica è:
1. Fondamenta teoriche (Pose Estimation, architetture video)
2. Sistemi esistenti (approcci ibridi + Transformer)
3. Dataset
4. Confronto comparativo → aggancio al Cap. 3

Questa è la struttura giusta per uno stato dell'arte.

### Sembra scritto da AI?

**Verdetto: parzialmente sì, ma si corregge facilmente.**

I segnali che un docente o un tool anti-AI potrebbe cogliere:

| Segnale | Dove | Esempio |
|---|---|---|
| **Aggettivi encomiastici vuoti** | Ovunque | "estremamente robusta", "fondamentale", "cruciale", "straordinariamente" |
| **Struttura a elenco troppo simmetrica** | §2.1.1 | Ogni paragrafo ha la stessa identica struttura: "Concetto X. Spiegazione. Limite." ripetuto meccanicamente |
| **Frasi-colla generiche** | Inizio paragrafi | "Un aspetto di particolare interesse riguarda..." — è un pattern classico di LLM |
| **Mancanza di voce personale** | Tutto il capitolo | Non c'è mai un momento in cui si percepisce che tu hai *letto* e *capito* quei paper. Sembra un riassunto automatico |

### Come migliorare (senza riscrivere tutto)

1. **Taglia gli aggettivi vuoti.** "Estremamente robusto" → "robusto". "Fondamentale" → rimuovilo o sostituisci con una frase che spieghi *perché* è importante. La regola è: **se togliendo l'aggettivo la frase perde informazione, tienilo; altrimenti eliminalo.**

2. **Aggiungi 2-3 commenti critici personali.** Per esempio dopo aver presentato CrimeNet:
   > *Tuttavia, l'accuratezza del 100% riportata suggerisce un possibile overfitting o una specificità eccessiva del benchmark, aspetto che limita la generalizzabilità del risultato.*
   
   Questo tipo di frase dimostra capacità critica e nessun LLM la mette spontaneamente.

3. **Varia la struttura dei paragrafi.** Non iniziare ogni sottosezione con la stessa formula "X propone Y. La pipeline è Z. I risultati sono W." Cambia l'ordine, parti dai risultati, parti dal problema.

4. **Riduci le frasi lunghissime.** Alcune frasi superano le 4 righe. Spezzale in due. Le frasi troppo lunghe e tecnicamente dense sono un segnale forte di generazione LLM.

---

## 3. Capitolo 3 — Metodologia e Architettura

### Qualità del contenuto: ⭐⭐⭐⭐⭐ Ottima

Questo capitolo è **il migliore dei due**. Il contenuto tecnico è preciso, le scelte sono ben motivate e tutto si aggancia alla letteratura. In particolare:

- La motivazione YOLO-Pose vs MediaPipe è argomentata con 3 punti tecnici solidi ✅
- La sezione sui keypoints COCO è chiara e dettagliata ✅
- La sliding window con (N, 30, 34) è definita bene ✅
- L'architettura LSTM (2 layer + 3 dense + dropout) è coerente ✅
- I dataset sono ben scelti e motivati ✅

### Sembra scritto da AI?

**Verdetto: meno del Cap. 2, ma ci sono ancora alcuni segnali.**

| Segnale | Dove | Come correggere |
|---|---|---|
| "immune alle interferenze visive" | §3.1.1, riga 42-43 | "Immune" è troppo assoluto. Usa "significativamente meno sensibile" |
| "cinematica pura dell'azione" | §3.1.1, riga 41 | Suona bene ma è usato 3 volte nel capitolo. Riducilo a 1 |
| "rumore di fondo" ripetuto | §3.1.1 e §3.2 | Usato sia nel Cap. 2 che nel Cap. 3. Varia |
| Paragrafi tutti della stessa lunghezza | §3.3 | Varia: un paragrafo corto (2-3 righe) + uno lungo rompe la monotonia |

### Problemi tecnici minori da correggere

1. **Riga 142-143** — *"I dati dunque subiscono un processo di normalizzazione"*: Nel tuo script 1, la normalizzazione avviene **durante l'estrazione** (dividi per frame_w e frame_h), non in un blocco separato di "pre-elaborazione temporale". Non è un errore grave ma se qualcuno legge il codice noterebbe l'incongruenza. Puoi riformulare dicendo che la normalizzazione avviene al momento dell'estrazione.

2. **§3.4, riga 236** — *"Il terzo e ultimo strato presenta una funzione di attivazione Softmax"*: Come discusso nella guida precedente, in PyTorch `CrossEntropyLoss` include il Softmax. Nel codice non avrai un layer Softmax esplicito. Questo **non è un problema per la tesi** — nella tesi descrivi l'architettura logica, non l'implementazione PyTorch. Ma tienilo a mente quando scrivi il Cap. 4.

3. **§3.5** — Dici di usare 3 dataset (RWF-2000, Real Life Violence, UCF-Crime). Alla fine li userai davvero tutti e tre? Se ne usi solo 1 o 2, aggiorna questa sezione.

---

## 4. Capitolo 4 — Struttura Proposta

### Titolo suggerito: *"Risultati Sperimentali"*

Ecco la struttura di sezioni e sottosezioni:

```latex
\chapter{Risultati Sperimentali}
\label{cap:cap4}

% Intro: cosa contiene il capitolo, come è organizzato

\section{Ambiente Sperimentale}
  \subsection{Hardware e Software}
    % PC usato per il training (GPU, RAM, OS)
    % Versioni: Python, PyTorch, Ultralytics
    % (Se testi su RPi: specifiche RPi 5)
    
  \subsection{Configurazione degli Iperparametri}
    % Tabella con tutti i parametri: 
    % lr, batch_size, epoche, hidden LSTM, dropout, stride, ecc.
    % Motiva brevemente le scelte (con riferimento al Cap. 3)

\section{Preparazione del Dataset}
  \subsection{Estrazione dei Keypoints}
    % Quanti video processati, tempo impiegato
    % Quante sequenze (30,34) generate in totale
    % Split train/test: quante sequenze per classe
    % Tabella riepilogativa: dataset | video | sequenze fight | sequenze no_fight
    
  \subsection{Analisi della Distribuzione dei Dati}
    % Il dataset è bilanciato? 
    % Istogrammi del numero di sequenze per video
    % Eventuali problemi (video troppo corti, persone non tracciate)

\section{Addestramento del Modello}
  \subsection{Processo di Training}
    % Grafico: Training Loss vs Validation Loss per epoca
    % Grafico: Training Accuracy vs Validation Accuracy per epoca
    % Discussione: convergenza, overfitting, early stopping
    % A quale epoca si è fermato l'early stopping?
    
  \subsection{Matrice di Confusione}
    % Matrice di confusione sul test set
    % Analisi errori: quali video vengono classificati male? Perché?

\section{Valutazione delle Prestazioni}
  \subsection{Metriche di Classificazione}
    % Tabella: Accuracy, Precision, Recall, F1-Score
    % Spiega ogni metrica brevemente
    % Confronta con i risultati della letteratura (tabella del Cap. 2)
    
  \subsection{Confronto con lo Stato dell'Arte}
    % Riprendi la tabella del Cap. 2 e aggiungi la tua riga
    % Commento critico: dove sei meglio, dove peggio, perché
    
  % (OPZIONALE) Se testi su RPi:
  \subsection{Prestazioni su Edge Device}
    % FPS su RPi 5
    % Latenza end-to-end
    % Confronto FPS: PC vs RPi 5

\section{Inferenza e Modulo di Rilevamento Armi}
  \subsection{Pipeline di Inferenza in Tempo Reale}
    % Come funziona lo script 3
    % Buffer circolare, sliding window in real-time
    % Screenshot o frame annotati con il risultato
    
  \subsection{Integrazione di YOLO Object Detection}
    % Come funziona il modulo armi (post-LSTM)
    % I 4 livelli di output
    % Esempi visivi: frame con classificazione finale
    % Limiti: falsi positivi YOLO su oggetti simili ad armi
```

---

## 5. Conviene prima addestrare o prima scrivere il Cap. 4?

**Risposta: prima addestra, poi scrivi.**

La ragione è semplice — il Capitolo 4 è fatto **quasi interamente di dati sperimentali** che non hai ancora:

| Sezione del Cap. 4 | Cosa ti serve | Ce l'hai? |
|---|---|---|
| Ambiente Sperimentale | Versioni software, hardware | ✅ Sì |
| Preparazione Dataset | Numero sequenze generate | ❌ Devi fare lo script 1 |
| Addestramento | Grafici loss, accuracy | ❌ Devi fare lo script 2 |
| Matrice di Confusione | Risultati su test set | ❌ Devi fare lo script 2 |
| Metriche | Accuracy, F1, ecc. | ❌ Devi fare lo script 2 |
| Confronto stato dell'arte | I tuoi numeri vs letteratura | ❌ Devi addestrare |
| Inferenza + Armi | Screenshot, demo | ❌ Devi fare lo script 3 |

L'unica sezione che puoi scrivere subito è **§4.1 Ambiente Sperimentale** (hardware, software, iperparametri scelti).

### Ordine di lavoro consigliato

```
1. Esegui lo script 1 sui video → genera i CSV
2. Scrivi e lancia lo script 2 → addestra LSTM → ottieni metriche e grafici
3. Scrivi il Cap. 4 con i dati reali in mano
4. Scrivi lo script 3 (inferenza) → ottieni screenshot per §4.5
5. Completa Cap. 4 con la parte inferenza
6. Scrivi Cap. 1 (Introduzione) e Conclusioni per ultimi
```

> [!TIP]
> **Il Cap. 1 (Introduzione) e le Conclusioni si scrivono SEMPRE per ultimi.** L'introduzione riassume il lavoro fatto, le conclusioni commentano i risultati. Entrambi richiedono che tutto il resto sia già finito.

### Cosa puoi fare ORA per il Cap. 4

Puoi preparare lo **scheletro LaTeX** del Cap. 4: crealo con tutte le `\section` e `\subsection` vuote, con commenti che indicano cosa ci andrà. Così quando avrai i dati, dovrai solo riempire. Questo è un approccio molto comune nelle tesi.
