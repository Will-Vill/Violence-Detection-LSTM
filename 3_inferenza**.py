"""
3_inferenza.py — Script di Inferenza per il Rilevamento di Risse

Modalità:
  python 3_inferenza.py --video percorso/al/video.mp4
  python 3_inferenza.py --webcam
  python 3_inferenza.py --test percorso/cartella_test/
"""

import argparse
import os
import sys
import time
import numpy as np
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
# CONFIGURAZIONE
# ══════════════════════════════════════════════════════════════════
MODELLO_YOLO = "yolo26n-pose.pt"
MODELLO_LSTM = "modello/modello_ottimizzato.pt"
SCALER_PATH  = "modello/scaler.pkl"

# Iperparametri (devono corrispondere a quelli dell'addestramento)
FINESTRA      = 45
STRIDE        = 15
NUM_FEATURES  = 69
HIDDEN_1      = 128
HIDDEN_2      = 64
DENSE_1       = 32
DENSE_2       = 16
NUM_CLASSI    = 2
DROPOUT       = 0.45

PULIZIA_FRAME = 90
SOGLIA_FIGHT  = 0.75
CONFERME_MIN  = 2


# ══════════════════════════════════════════════════════════════════
# MODELLO LSTM
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
        x = x[:, -1, :]
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout4(x)
        x = self.fc3(x)
        return x


# ══════════════════════════════════════════════════════════════════
# CONNESSIONI SCHELETRO COCO
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

    def _calcola_features(self, coords_list, centri_list, pid):
        coords = np.array(coords_list)

        # Velocità frame-to-frame
        velocita = np.diff(coords, axis=0)
        velocita = np.vstack([np.zeros((1, 34)), velocita])

        # Distanza dal vicino più prossimo
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

        distanze = np.array(distanze).reshape(-1, 1)

        return np.hstack([coords, velocita, distanze])

    def _classifica(self, features):
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

    def processa_frame(self, frame):
        self.frame_count += 1
        h, w = frame.shape[:2]

        risultati = self.yolo.track(frame, persist=True, verbose=False)
        r = risultati[0]

        centri_frame = {}
        persone_frame = set()

        if (r.keypoints is not None and r.boxes is not None
                and r.boxes.id is not None):

            kp_persone = r.keypoints.xy.cpu().numpy()
            id_persone = r.boxes.id.cpu().numpy().astype(int)
            boxes      = r.boxes.xyxy.cpu().numpy().astype(int)

            for i, pid in enumerate(id_persone):
                kp = kp_persone[i]
                cx = np.mean(kp[:, 0]) / w
                cy = np.mean(kp[:, 1]) / h
                centri_frame[pid] = (cx, cy)
                persone_frame.add(pid)

            for i, pid in enumerate(id_persone):
                kp = kp_persone[i]

                coords_norm = np.empty(34)
                for j in range(17):
                    coords_norm[j * 2]     = round(kp[j][0] / w, 4)
                    coords_norm[j * 2 + 1] = round(kp[j][1] / h, 4)

                if pid not in self.buffer_persone:
                    self.buffer_persone[pid] = {
                        'coords':       [],
                        'centri':       [],
                        'predizione':   None,
                        'probabilita':  0.0,
                        'contatore_fight': 0,
                        'ultimo_frame': self.frame_count,
                    }

                buf = self.buffer_persone[pid]
                buf['coords'].append(coords_norm)
                buf['centri'].append(centri_frame.copy())
                buf['ultimo_frame'] = self.frame_count

                if len(buf['coords']) >= FINESTRA:
                    seq_coords = buf['coords'][-FINESTRA:]
                    seq_centri = buf['centri'][-FINESTRA:]

                    features = self._calcola_features(seq_coords, seq_centri, pid)
                    classe_raw, prob = self._classifica(features)

                    if prob >= SOGLIA_FIGHT:
                        buf['contatore_fight'] += 1
                    else:
                        buf['contatore_fight'] = 0

                    if buf['contatore_fight'] >= CONFERME_MIN:
                        buf['predizione'] = 1
                    else:
                        buf['predizione'] = 0
                    buf['probabilita'] = prob

                    buf['coords'] = buf['coords'][STRIDE:]
                    buf['centri'] = buf['centri'][STRIDE:]

            for i, pid in enumerate(id_persone):
                box = boxes[i]
                kp  = kp_persone[i]
                buf = self.buffer_persone.get(pid, {})
                pred = buf.get('predizione')
                prob = buf.get('probabilita', 0.0)

                if pred == 1:
                    colore   = (0, 0, 255)
                    etichetta = f"FIGHT {prob*100:.0f}%"
                elif pred == 0:
                    colore   = (0, 220, 0)
                    etichetta = "OK"
                else:
                    colore   = (180, 180, 180)
                    etichetta = "..."

                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), colore, 2)

                testo = f"ID:{pid} {etichetta}"
                (tw, th), _ = cv2.getTextSize(testo, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame,
                              (box[0], box[1] - th - 10),
                              (box[0] + tw + 5, box[1]),
                              colore, -1)
                cv2.putText(frame, testo,
                            (box[0] + 2, box[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                self._disegna_scheletro(frame, kp, colore)

        da_rimuovere = [
            pid for pid, buf in self.buffer_persone.items()
            if self.frame_count - buf['ultimo_frame'] > PULIZIA_FRAME
        ]
        for pid in da_rimuovere:
            del self.buffer_persone[pid]

        self.rissa_rilevata = any(
            buf.get('predizione') == 1
            for buf in self.buffer_persone.values()
        )

        if self.rissa_rilevata:
            cv2.rectangle(frame, (0, 0), (w, 50), (0, 0, 200), -1)
            cv2.putText(frame, "RISSA RILEVATA",
                        (15, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
        else:
            cv2.rectangle(frame, (0, 0), (w, 50), (0, 130, 0), -1)
            cv2.putText(frame, "SITUAZIONE NORMALE",
                        (15, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)

        return frame, self.rissa_rilevata


    def esegui_video(self, percorso_video, salva_output=True):
        print(f"Apertura video: {percorso_video}")

        cap = cv2.VideoCapture(percorso_video)
        if not cap.isOpened():
            sys.exit(f"ERRORE: impossibile aprire {percorso_video}")

        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        totale = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {w}x{h} @ {fps_video:.0f} FPS — {totale} frame\n")

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


    def esegui_webcam(self):
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


    SOGLIA_VOTO_TEST = 0.15

    def _classifica_video_test(self, percorso_video):
        self.buffer_persone = {}
        self.frame_count = 0

        cap = cv2.VideoCapture(percorso_video)
        if not cap.isOpened():
            return None, 0, 0, 0.0

        h, w = None, None
        predizioni_finestre = []

        buffer_test = {}

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            self.frame_count += 1
            if h is None:
                h, w = frame.shape[:2]

            risultati = self.yolo.track(frame, persist=True, verbose=False)
            r = risultati[0]

            centri_frame = {}

            if (r.keypoints is not None and r.boxes is not None
                    and r.boxes.id is not None):

                kp_persone = r.keypoints.xy.cpu().numpy()
                id_persone = r.boxes.id.cpu().numpy().astype(int)

                for i, pid in enumerate(id_persone):
                    kp = kp_persone[i]
                    cx = np.mean(kp[:, 0]) / w
                    cy = np.mean(kp[:, 1]) / h
                    centri_frame[pid] = (cx, cy)

                for i, pid in enumerate(id_persone):
                    kp = kp_persone[i]

                    coords_norm = np.empty(34)
                    for j in range(17):
                        coords_norm[j * 2]     = round(kp[j][0] / w, 4)
                        coords_norm[j * 2 + 1] = round(kp[j][1] / h, 4)

                    if pid not in buffer_test:
                        buffer_test[pid] = {
                            'coords': [],
                            'centri': [],
                            'ultimo_frame': self.frame_count,
                        }

                    buf = buffer_test[pid]
                    buf['coords'].append(coords_norm)
                    buf['centri'].append(centri_frame.copy())
                    buf['ultimo_frame'] = self.frame_count

                    if len(buf['coords']) >= FINESTRA:
                        seq_coords = buf['coords'][-FINESTRA:]
                        seq_centri = buf['centri'][-FINESTRA:]

                        features = self._calcola_features(seq_coords, seq_centri, pid)
                        classe_raw, prob = self._classifica(features)

                        predizioni_finestre.append(classe_raw)

                        buf['coords'] = buf['coords'][STRIDE:]
                        buf['centri'] = buf['centri'][STRIDE:]

            da_rimuovere = [
                pid for pid, buf in buffer_test.items()
                if self.frame_count - buf['ultimo_frame'] > PULIZIA_FRAME
            ]
            for pid in da_rimuovere:
                del buffer_test[pid]

        cap.release()

        n_finestre = len(predizioni_finestre)
        if n_finestre == 0:
            return 0, 0, 0, 0.0

        n_fight = sum(predizioni_finestre)
        perc_fight = n_fight / n_finestre

        predizione = 1 if perc_fight >= self.SOGLIA_VOTO_TEST else 0

        return predizione, n_finestre, n_fight, perc_fight

    def esegui_test(self, cartella_test):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import classification_report, confusion_matrix

        print(f"{'='*60}")
        print(f" VALUTAZIONE TEST SET: {cartella_test}")
        print(f" Soglia voto: {self.SOGLIA_VOTO_TEST*100:.0f}% delle finestre")
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

                pred, n_fin, n_fight, perc = self._classifica_video_test(percorso)

                if pred is None:
                    print(f"    [{idx}/{len(video_files)}] {nome_video} → ERRORE apertura")
                    continue

                y_veri.append(etichetta_vera)
                y_predetti.append(pred)

                esito = "✓" if pred == etichetta_vera else "✗"
                nome_p = "FIGHT" if pred == 1 else "NO_FIGHT"
                print(f"    [{idx}/{len(video_files)}] {nome_video}"
                      f"  →  {nome_p}  {esito}  "
                      f"({n_fight}/{n_fin} finestre fight = {perc*100:.1f}%)")

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

        os.makedirs("grafici", exist_ok=True)
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
