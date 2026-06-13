"""
3_inferenza.py — Script di Inferenza per il Rilevamento di Risse

Modalità di utilizzo:
  python 3_inferenza.py --video percorso/al/video.mp4      → Analizza un file video
  python 3_inferenza.py --webcam                            → Inferenza in tempo reale
  python 3_inferenza.py --test percorso/cartella_test/      → Valuta su un Test Set

Requisiti:
  - modello/lstm_risse.pt   (modello LSTM addestrato)
  - modello/scaler.pkl      (StandardScaler salvato dal training)
  - yolo26n-pose.pt         (modello YOLO-Pose)
"""

import argparse
import os
import sys
import time
import numpy as np
import os
import platform
if platform.system() == "Linux":
    os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
import torch
import torch.nn as nn
import joblib
from ultralytics import YOLO
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════
# CONFIGURAZIONE — deve coincidere con il modello addestrato
# ══════════════════════════════════════════════════════════════════
MODELLO_YOLO = "yolo26x-pose.pt"
MODELLO_LSTM = "modello/lstm_risse.pt"
SCALER_PATH  = "modello/scaler.pkl"

# Iperparametri (devono corrispondere a quelli dell'addestramento)
FINESTRA      = 45       # Finestra temporale (45 fotogrammi = 1.5s a 30fps)
STRIDE        = 15       # Passo di scorrimento della sliding window
NUM_FEATURES  = 69       # 34 coordinate + 34 velocità + 1 distanza
HIDDEN_1      = 128
HIDDEN_2      = 64
DENSE_1       = 32
DENSE_2       = 16
NUM_CLASSI    = 2
DROPOUT       = 0.45     # Ignorato in eval mode, serve solo per caricare i pesi

PULIZIA_FRAME = 90       # Rimuovi persone non viste da N frame


# ══════════════════════════════════════════════════════════════════
# MODELLO LSTM — struttura identica al file di addestramento (v2)
# ══════════════════════════════════════════════════════════════════
class LSTMClassificatore(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm1   = nn.LSTM(input_size=NUM_FEATURES, hidden_size=HIDDEN_1, batch_first=True)
        self.dropout1 = nn.Dropout(DROPOUT)
        self.lstm2   = nn.LSTM(input_size=HIDDEN_1, hidden_size=HIDDEN_2, batch_first=True)
        self.dropout2 = nn.Dropout(DROPOUT)
        self.fc1     = nn.Linear(HIDDEN_2, DENSE_1)
        self.dropout3 = nn.Dropout(DROPOUT)
        self.fc2     = nn.Linear(DENSE_1, DENSE_2)
        self.dropout4 = nn.Dropout(DROPOUT)
        self.fc3     = nn.Linear(DENSE_2, NUM_CLASSI)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = x[:, -1, :]                   # Ultimo step temporale
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout4(x)
        x = self.fc3(x)
        return x


# ══════════════════════════════════════════════════════════════════
# CONNESSIONI SCHELETRO COCO (per il disegno)
# ══════════════════════════════════════════════════════════════════
CONNESSIONI_COCO = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # Testa
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # Braccia
    (5, 11), (6, 12), (11, 12),                  # Torso
    (11, 13), (13, 15), (12, 14), (14, 16)       # Gambe
]


# ══════════════════════════════════════════════════════════════════
# CLASSE PRINCIPALE
# ══════════════════════════════════════════════════════════════════
class InferenzaRisse:

    def __init__(self):
        # --- Dispositivo ---
        self.dispositivo = torch.device(
            'cuda' if torch.cuda.is_available()
            else 'mps' if torch.backends.mps.is_available()
            else 'cpu'
        )
        print(f"Dispositivo: {self.dispositivo}")

        # --- Carica YOLO-Pose ---
        if not os.path.exists(MODELLO_YOLO):
            sys.exit(f"ERRORE: modello YOLO non trovato → {MODELLO_YOLO}")
        print("Caricamento YOLO-Pose...")
        self.yolo = YOLO(MODELLO_YOLO)

        # --- Carica LSTM ---
        if not os.path.exists(MODELLO_LSTM):
            sys.exit(f"ERRORE: modello LSTM non trovato → {MODELLO_LSTM}")
        print("Caricamento modello LSTM...")
        self.modello = LSTMClassificatore().to(self.dispositivo)
        self.modello.load_state_dict(
            torch.load(MODELLO_LSTM, map_location=self.dispositivo, weights_only=True)
        )
        self.modello.eval()

        # --- Carica StandardScaler ---
        if not os.path.exists(SCALER_PATH):
            sys.exit(f"ERRORE: scaler non trovato → {SCALER_PATH}")
        print("Caricamento StandardScaler...")
        self.scaler = joblib.load(SCALER_PATH)

        # --- Buffer per le persone tracciate ---
        self.buffer_persone = {}
        self.frame_count = 0
        self.rissa_rilevata = False

        print("Sistema pronto!\n")

    # ──────────────────────────────────────────────────────────────
    # Calcolo features (identico al training)
    # ──────────────────────────────────────────────────────────────
    def _calcola_features(self, coords_list, centri_list, pid):
        """
        Calcola le 69 features per una sequenza di FINESTRA frame.
        coords_list : lista di array (34,) — coordinate normalizzate
        centri_list : lista di dict {pid: (cx, cy)} per ciascun frame
        pid         : id della persona da analizzare
        """
        coords = np.array(coords_list)            # (FINESTRA, 34)

        # Velocità frame-to-frame (identico al training)
        velocita = np.diff(coords, axis=0)
        velocita = np.vstack([np.zeros((1, 34)), velocita])   # (FINESTRA, 34)

        # Distanza dal vicino più prossimo (identico al training)
        distanze = []
        for centri in centri_list:
            if pid not in centri or len(centri) < 2:
                distanze.append(1.0)
            else:
                centro = centri[pid]
                min_dist = min(
                    np.sqrt((centro[0] - c[0])**2 + (centro[1] - c[1])**2)
                    for p, c in centri.items() if p != pid
                )
                distanze.append(min_dist)

        distanze = np.array(distanze).reshape(-1, 1)  # (FINESTRA, 1)

        # Concatena: coordinate(34) + velocità(34) + distanza(1) = 69
        return np.hstack([coords, velocita, distanze])

    # ──────────────────────────────────────────────────────────────
    # Classificazione di una sequenza
    # ──────────────────────────────────────────────────────────────
    def _classifica(self, features):
        """
        Classifica una sequenza (FINESTRA, 69).
        Ritorna (classe, probabilità_fight).
        """
        features_flat = features.reshape(-1, NUM_FEATURES)
        features_norm = self.scaler.transform(features_flat)
        features_norm = features_norm.reshape(1, FINESTRA, NUM_FEATURES)

        tensor = torch.tensor(features_norm, dtype=torch.float32).to(self.dispositivo)
        with torch.no_grad():
            output = self.modello(tensor)
            prob   = torch.softmax(output, dim=1)
            classe = output.argmax(1).item()
            prob_fight = prob[0][1].item()

        return classe, prob_fight

    # ──────────────────────────────────────────────────────────────
    # Disegna lo scheletro
    # ──────────────────────────────────────────────────────────────
    def _disegna_scheletro(self, frame, kp, colore):
        for (a, b) in CONNESSIONI_COCO:
            pt1 = (int(kp[a][0]), int(kp[a][1]))
            pt2 = (int(kp[b][0]), int(kp[b][1]))
            if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                cv2.line(frame, pt1, pt2, colore, 2)
        for j in range(17):
            x, y = int(kp[j][0]), int(kp[j][1])
            if x > 0 and y > 0:
                cv2.circle(frame, (x, y), 4, colore, -1)

    # ──────────────────────────────────────────────────────────────
    # Processa un singolo frame
    # ──────────────────────────────────────────────────────────────
    def processa_frame(self, frame):
        """
        Pipeline completa per un frame:
        1. YOLO-Pose → detection + tracking
        2. Aggiorna i buffer per ciascuna persona
        3. Classifica le sequenze complete
        4. Annota il frame

        Ritorna: (frame_annotato, rissa_rilevata)
        """
        self.frame_count += 1
        h, w = frame.shape[:2]

        # 1. YOLO-Pose detection + tracking
        risultati = self.yolo.track(frame, persist=True, verbose=False)
        r = risultati[0]

        centri_frame = {}     # {pid: (cx, cy)} — centri di massa normalizzati
        persone_frame = set()

        if (r.keypoints is not None and r.boxes is not None
                and r.boxes.id is not None):

            kp_persone = r.keypoints.xy.cpu().numpy()     # (N, 17, 2) pixel
            id_persone = r.boxes.id.cpu().numpy().astype(int)
            boxes      = r.boxes.xyxy.cpu().numpy().astype(int)

            # Centri di massa normalizzati (per calcolo distanza)
            for i, pid in enumerate(id_persone):
                kp = kp_persone[i]
                cx = np.mean(kp[:, 0]) / w
                cy = np.mean(kp[:, 1]) / h
                centri_frame[pid] = (cx, cy)
                persone_frame.add(pid)

            # 2. Aggiorna i buffer per ciascuna persona
            for i, pid in enumerate(id_persone):
                kp = kp_persone[i]

                # Coordinate normalizzate 0-1 (come lo script di estrazione)
                coords_norm = np.empty(34)
                for j in range(17):
                    coords_norm[j * 2]     = round(kp[j][0] / w, 4)
                    coords_norm[j * 2 + 1] = round(kp[j][1] / h, 4)

                # Inizializza buffer se persona nuova
                if pid not in self.buffer_persone:
                    self.buffer_persone[pid] = {
                        'coords':       [],
                        'centri':       [],
                        'predizione':   None,
                        'probabilita':  0.0,
                        'ultimo_frame': self.frame_count,
                    }

                buf = self.buffer_persone[pid]
                buf['coords'].append(coords_norm)
                buf['centri'].append(centri_frame.copy())
                buf['ultimo_frame'] = self.frame_count

                # 3. Classifica quando la finestra è piena
                if len(buf['coords']) >= FINESTRA:
                    seq_coords = buf['coords'][-FINESTRA:]
                    seq_centri = buf['centri'][-FINESTRA:]

                    features = self._calcola_features(seq_coords, seq_centri, pid)
                    classe, prob = self._classifica(features)

                    buf['predizione']  = classe
                    buf['probabilita'] = prob

                    # Sliding window: avanza di STRIDE
                    buf['coords'] = buf['coords'][STRIDE:]
                    buf['centri'] = buf['centri'][STRIDE:]

            # 4. Annota il frame
            for i, pid in enumerate(id_persone):
                box = boxes[i]
                kp  = kp_persone[i]
                buf = self.buffer_persone.get(pid, {})
                pred = buf.get('predizione')
                prob = buf.get('probabilita', 0.0)

                if pred == 1:                               # FIGHT
                    colore   = (0, 0, 255)
                    etichetta = f"FIGHT {prob*100:.0f}%"
                elif pred == 0:                             # NO FIGHT
                    colore   = (0, 220, 0)
                    etichetta = "OK"
                else:                                       # In attesa
                    colore   = (180, 180, 180)
                    etichetta = "..."

                # Bounding box
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), colore, 2)

                # Etichetta con sfondo
                testo = f"ID:{pid} {etichetta}"
                (tw, th), _ = cv2.getTextSize(testo, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame,
                              (box[0], box[1] - th - 10),
                              (box[0] + tw + 5, box[1]),
                              colore, -1)
                cv2.putText(frame, testo,
                            (box[0] + 2, box[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Scheletro
                self._disegna_scheletro(frame, kp, colore)

        # Pulizia: rimuovi persone non viste da PULIZIA_FRAME frame
        da_rimuovere = [
            pid for pid, buf in self.buffer_persone.items()
            if self.frame_count - buf['ultimo_frame'] > PULIZIA_FRAME
        ]
        for pid in da_rimuovere:
            del self.buffer_persone[pid]

        # Stato globale: c'è una rissa?
        self.rissa_rilevata = any(
            buf.get('predizione') == 1
            for buf in self.buffer_persone.values()
        )

        # Banner superiore
        if self.rissa_rilevata:
            cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 200), -1)
            cv2.putText(frame, "RISSA RILEVATA",
                        (15, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        else:
            cv2.rectangle(frame, (0, 0), (w, 50), (0, 130, 0), -1)
            cv2.putText(frame, "SITUAZIONE NORMALE",
                        (15, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

        return frame, self.rissa_rilevata


    # ══════════════════════════════════════════════════════════════
    # MODALITÀ VIDEO
    # ══════════════════════════════════════════════════════════════
    def esegui_video(self, percorso_video, salva_output=True):
        """Analizza un file video e mostra i risultati a schermo."""
        print(f"Apertura video: {percorso_video}")

        cap = cv2.VideoCapture(percorso_video)
        if not cap.isOpened():
            sys.exit(f"ERRORE: impossibile aprire {percorso_video}")

        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        totale = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {w}x{h} @ {fps_video:.0f} FPS — {totale} frame\n")

        # Writer per il video annotato
        writer = None
        nome_output = None
        if salva_output:
            nome_output = os.path.splitext(percorso_video)[0] + "_inferenza.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(nome_output, fourcc, fps_video, (w, h))

        processati = 0
        t0 = time.time()

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame_ann, _ = self.processa_frame(frame)

            if writer:
                writer.write(frame_ann)

            cv2.imshow("Inferenza Risse", frame_ann)
            processati += 1

            if processati % 100 == 0:
                fps = processati / (time.time() - t0)
                print(f"  Frame {processati}/{totale} | FPS: {fps:.1f}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        durata = time.time() - t0
        fps_medio = processati / durata if durata > 0 else 0

        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

        print(f"\nCompletato: {processati} frame in {durata:.1f}s ({fps_medio:.1f} FPS medio)")
        if nome_output:
            print(f"Video annotato salvato: {nome_output}")


    # ══════════════════════════════════════════════════════════════
    # MODALITÀ WEBCAM
    # ══════════════════════════════════════════════════════════════
    def esegui_webcam(self):
        """Inferenza in tempo reale dalla webcam."""
        print("Avvio webcam ... (premi 'q' per uscire)\n")

        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            sys.exit("ERRORE: impossibile aprire la webcam")

        fps_cnt = 0
        fps_t0  = time.time()
        fps_val = 0.0

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame_ann, _ = self.processa_frame(frame)

            # Calcolo FPS in tempo reale
            fps_cnt += 1
            dt = time.time() - fps_t0
            if dt >= 1.0:
                fps_val = fps_cnt / dt
                fps_cnt = 0
                fps_t0  = time.time()

            cv2.putText(frame_ann, f"FPS: {fps_val:.0f}",
                        (frame_ann.shape[1] - 140, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Inferenza Risse - Tempo Reale", frame_ann)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


    # ══════════════════════════════════════════════════════════════
    # MODALITÀ TEST SET
    # ══════════════════════════════════════════════════════════════
    def esegui_test(self, cartella_test):
        """
        Valuta il modello su un Test Set.

        Struttura attesa della cartella:
          cartella_test/
          ├── fight/
          │   ├── video1.mp4
          │   └── video2.avi
          └── no_fight/
              ├── video3.mp4
              └── ...

        Per ogni video, il sistema classifica tutte le sequenze
        e decide con majority voting se il video contiene una rissa.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import classification_report, confusion_matrix

        print(f"{'='*60}")
        print(f" VALUTAZIONE TEST SET: {cartella_test}")
        print(f"{'='*60}\n")

        y_veri     = []
        y_predetti = []

        for categoria in ["fight", "no_fight"]:
            cartella = os.path.join(cartella_test, categoria)
            if not os.path.exists(cartella):
                print(f"  Cartella non trovata: {cartella}")
                continue

            etichetta_vera = 1 if categoria == "fight" else 0
            video_files = sorted([
                f for f in os.listdir(cartella)
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
            ])

            nome_cat = "Rissa" if categoria == "fight" else "Normale"
            print(f"\n  {nome_cat} ({categoria}) — {len(video_files)} video")

            for idx, nome_video in enumerate(video_files, 1):
                percorso = os.path.join(cartella, nome_video)

                # Reset completo del buffer per ogni video
                self.buffer_persone = {}
                self.frame_count = 0

                cap = cv2.VideoCapture(percorso)
                if not cap.isOpened():
                    print(f"    [{idx}/{len(video_files)}] {nome_video} → ERRORE apertura")
                    continue

                predizioni_video = []

                while cap.isOpened():
                    ok, frame = cap.read()
                    if not ok:
                        break
                    self.processa_frame(frame)

                    # Raccogli le predizioni di tutte le persone
                    for pid, buf in self.buffer_persone.items():
                        if buf['predizione'] is not None:
                            predizioni_video.append(buf['predizione'])

                cap.release()

                # Majority voting
                if len(predizioni_video) > 0:
                    predizione = 1 if sum(predizioni_video) > len(predizioni_video) / 2 else 0
                else:
                    predizione = 0

                y_veri.append(etichetta_vera)
                y_predetti.append(predizione)

                esito = "✓" if predizione == etichetta_vera else "✗"
                nome_p = "FIGHT" if predizione == 1 else "NO_FIGHT"
                n_seq  = len(predizioni_video)
                print(f"    [{idx}/{len(video_files)}] {nome_video}"
                      f"  →  {nome_p}  {esito}  ({n_seq} sequenze)")

        # ── Metriche ──
        if len(y_veri) == 0:
            print("\nNessun video processato.")
            return

        y_veri     = np.array(y_veri)
        y_predetti = np.array(y_predetti)

        print(f"\n{'='*60}")
        print(" METRICHE TEST SET")
        print(f"{'='*60}\n")
        print(classification_report(
            y_veri, y_predetti, target_names=["no_fight", "fight"]
        ))

        cm = confusion_matrix(y_veri, y_predetti)
        print("Matrice di confusione:")
        print(cm)

        # Salva matrice di confusione
        os.makedirs("grafici", exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["no_fight", "fight"])
        ax.set_yticklabels(["no_fight", "fight"])
        ax.set_xlabel("Predetto")
        ax.set_ylabel("Reale")
        ax.set_title("Matrice di Confusione — Test Set")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha='center', va='center', fontsize=20, fontweight='bold')
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig("grafici/confusion_matrix_test.png", dpi=150)
        plt.close()
        print("\nGrafico salvato: grafici/confusion_matrix_test.png")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Inferenza per il Rilevamento Automatico di Risse"
    )

    gruppo = parser.add_mutually_exclusive_group(required=True)
    gruppo.add_argument("--video",  type=str,
                        help="Percorso di un file video da analizzare")
    gruppo.add_argument("--webcam", action="store_true",
                        help="Inferenza in tempo reale dalla webcam")
    gruppo.add_argument("--test",   type=str,
                        help="Cartella di test con sottocartelle fight/ e no_fight/")

    parser.add_argument("--no-save", action="store_true",
                        help="Non salvare il video annotato (solo modalità --video)")

    args = parser.parse_args()

    sistema = InferenzaRisse()

    if args.video:
        sistema.esegui_video(args.video, salva_output=not args.no_save)
    elif args.webcam:
        sistema.esegui_webcam()
    elif args.test:
        sistema.esegui_test(args.test)


if __name__ == "__main__":
    main()
