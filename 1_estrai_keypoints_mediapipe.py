import os
import csv
import glob
import cv2
import time
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# CONFIGURAZIONE
# YOLO viene usato SOLO per il rilevamento e il tracking delle persone
# (bounding box + ID). La stima della posa è delegata a MediaPipe.
MODELLO_DETECTION = "yolo26n-pose.pt"
CARTELLA_DATASET = "datasets"
CARTELLA_OUTPUT = "output_mediapipe"

# Mapping da MediaPipe (33 landmark) → 17 keypoints COCO
# Questo assicura che l'output CSV abbia lo stesso formato
# dello script YOLO-Pose, così lo script 2 funziona senza modifiche.
MAPPING_MP_A_COCO = [
    0,   # 0  naso        → MP nose
    2,   # 1  occhio_sx   → MP left_eye
    5,   # 2  occhio_dx   → MP right_eye
    7,   # 3  orecchio_sx → MP left_ear
    8,   # 4  orecchio_dx → MP right_ear
    11,  # 5  spalla_sx   → MP left_shoulder
    12,  # 6  spalla_dx   → MP right_shoulder
    13,  # 7  gomito_sx   → MP left_elbow
    14,  # 8  gomito_dx   → MP right_elbow
    15,  # 9  polso_sx    → MP left_wrist
    16,  # 10 polso_dx    → MP right_wrist
    23,  # 11 anca_sx     → MP left_hip
    24,  # 12 anca_dx     → MP right_hip
    25,  # 13 ginocchio_sx→ MP left_knee
    26,  # 14 ginocchio_dx→ MP right_knee
    27,  # 15 caviglia_sx → MP left_ankle
    28,  # 16 caviglia_dx → MP right_ankle
]

NOMI_KP = [
    "naso", "occhio_sx", "occhio_dx", "orecchio_sx", "orecchio_dx",
    "spalla_sx", "spalla_dx", "gomito_sx", "gomito_dx", "polso_sx", "polso_dx",
    "anca_sx", "anca_dx", "ginocchio_sx", "ginocchio_dx", "caviglia_sx", "caviglia_dx"
]


def estrazione_video_csv(modello_yolo, pose_mp, percorso_video, percorso_csv):
    """
    Estrae i keypoints da un video usando MediaPipe per la posa.
    YOLO viene usato solo per rilevare e tracciare le persone (bounding box + ID).
    Per ogni persona rilevata, il ritaglio viene passato a MediaPipe Pose
    che stima i 33 landmark, poi mappati ai 17 keypoints COCO.
    """
    video = cv2.VideoCapture(percorso_video)
    if not video.isOpened():
        return 0

    frame_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    header = ["frame", "id_persona"]
    for kp in NOMI_KP:
        header.append(f"{kp}_x")
        header.append(f"{kp}_y")

    contatore_frame = 0

    with open(percorso_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        while video.isOpened():
            ok, frame = video.read()
            if not ok:
                break

            # --- PASSO 1: YOLO rileva e traccia le persone ---
            risultati = modello_yolo.track(frame, persist=True, verbose=False, device=0)
            r = risultati[0]

            if r.boxes is not None and r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()      # [x1, y1, x2, y2]
                id_persone = r.boxes.id.cpu().numpy().astype(int)

                for i, pid in enumerate(id_persone):
                    x1, y1, x2, y2 = boxes[i].astype(int)

                    # Limita le coordinate ai bordi del frame
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(frame_w, x2)
                    y2 = min(frame_h, y2)

                    crop_w = x2 - x1
                    crop_h = y2 - y1

                    # Salta ritagli troppo piccoli
                    if crop_w < 20 or crop_h < 20:
                        continue

                    # --- PASSO 2: Ritaglia la persona e passa a MediaPipe ---
                    ritaglio = frame[y1:y2, x1:x2]
                    ritaglio_rgb = cv2.cvtColor(ritaglio, cv2.COLOR_BGR2RGB)

                    risultato_mp = pose_mp.process(ritaglio_rgb)

                    riga = [contatore_frame, pid]

                    if risultato_mp.pose_landmarks:
                        landmarks = risultato_mp.pose_landmarks.landmark

                        for indice_coco in range(17):
                            indice_mp = MAPPING_MP_A_COCO[indice_coco]
                            lm = landmarks[indice_mp]

                            # Converti da coordinate relative al ritaglio
                            # a coordinate normalizzate rispetto al frame intero
                            x_frame = (x1 + lm.x * crop_w) / frame_w
                            y_frame = (y1 + lm.y * crop_h) / frame_h
                            riga.append(round(x_frame, 4))
                            riga.append(round(y_frame, 4))
                    else:
                        # MediaPipe non ha trovato la posa: scrivi zeri
                        for _ in range(17):
                            riga.append(0.0)
                            riga.append(0.0)

                    writer.writerow(riga)

            contatore_frame += 1

    video.release()
    return contatore_frame



def main():
    splits = ["train", "val"]
    categorie = ["fight", "no_fight"]

    # Crea le cartelle di input e output
    for split in splits:
        for categoria in categorie:
            os.makedirs(os.path.join(CARTELLA_DATASET, split, categoria), exist_ok=True)
            os.makedirs(os.path.join(CARTELLA_OUTPUT, split, categoria), exist_ok=True)

    # Carica YOLO solo per il rilevamento e tracking delle persone
    modello_yolo = YOLO(MODELLO_DETECTION)

    # Inizializza MediaPipe Pose
    # static_image_mode=True perché ogni ritaglio è indipendente
    # model_complexity=1 per un buon compromesso precisione/velocità
    pose_mp = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=0.5
    )

    numero_video_totali = 0

    for split in splits:
        print(f"\n{'='*50}")
        print(f" Split: {split.upper()} — Estrazione con MediaPipe")
        print(f"{'='*50}")

        for categoria in categorie:
            cartella_video = os.path.join(CARTELLA_DATASET, split, categoria)
            lista_video = sorted(glob.glob(os.path.join(cartella_video, "*.*")))
            lista_video = [v for v in lista_video if v.endswith((".mp4", ".avi", ".mov", ".mkv"))]

            if not lista_video:
                print(f"\n  Nessun video in {cartella_video}/")
                continue

            etichetta = "Rissa" if categoria == "fight" else "Non rissa"
            print(f"\n  {etichetta} ({categoria}) — {len(lista_video)} video")

            for indice, percorso_video in enumerate(lista_video, 1):
                nome_file = os.path.basename(percorso_video)
                print(f"    [{indice}/{len(lista_video)}] {nome_file} ... ", end="", flush=True)

                tempo_inizio = time.time()
                nome_csv = os.path.splitext(nome_file)[0] + ".csv"
                percorso_csv = os.path.join(CARTELLA_OUTPUT, split, categoria, nome_csv)

                numero_frame = estrazione_video_csv(modello_yolo, pose_mp, percorso_video, percorso_csv)

                if numero_frame > 0:
                    print(f"Successo: {numero_frame} frame | {time.time()-tempo_inizio:.1f}s")
                else:
                    print(f"Errore | {time.time()-tempo_inizio:.1f}s")

                numero_video_totali += 1

    pose_mp.close()

    print(f"\n{'='*50}")
    print(f" Video processati: {numero_video_totali}")
    print(f" Output in: {CARTELLA_OUTPUT}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
