import os
import csv
import glob
import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

# CONFIGURAZIONE
MODELLO_MP = "pose_landmarker.task"
CARTELLA_DATASET = "datasets/RWF2000_RealLifeViolence"
CARTELLA_OUTPUT = "output_mediapipe"
MAX_PERSONE = 5  # numero massimo di persone da rilevare per frame

# Mapping da MediaPipe (33 landmark) ai 17 keypoints COCO
MAPPING_MP_A_COCO = [
    0,   # 0  naso
    2,   # 1  occhio_sx
    5,   # 2  occhio_dx
    7,   # 3  orecchio_sx
    8,   # 4  orecchio_dx
    11,  # 5  spalla_sx
    12,  # 6  spalla_dx
    13,  # 7  gomito_sx
    14,  # 8  gomito_dx
    15,  # 9  polso_sx
    16,  # 10 polso_dx
    23,  # 11 anca_sx
    24,  # 12 anca_dx
    25,  # 13 ginocchio_sx
    26,  # 14 ginocchio_dx
    27,  # 15 caviglia_sx
    28,  # 16 caviglia_dx
]

NOMI_KP = [
    "naso", "occhio_sx", "occhio_dx", "orecchio_sx", "orecchio_dx",
    "spalla_sx", "spalla_dx", "gomito_sx", "gomito_dx", "polso_sx", "polso_dx",
    "anca_sx", "anca_dx", "ginocchio_sx", "ginocchio_dx", "caviglia_sx", "caviglia_dx"
]


def calcola_centro(landmarks):
    """Calcola il centro di una persona come media delle coordinate delle anche."""
    anca_sx = landmarks[23]
    anca_dx = landmarks[24]
    centro_x = (anca_sx.x + anca_dx.x) / 2
    centro_y = (anca_sx.y + anca_dx.y) / 2
    return centro_x, centro_y


def estrazione_video_csv(detector, percorso_video, percorso_csv):
    """Estrae i keypoints da un video usando MediaPipe."""
    video = cv2.VideoCapture(percorso_video)
    if not video.isOpened():
        return 0

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

            # Converti BGR (OpenCV) → RGB (MediaPipe)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Crea un oggetto Image di MediaPipe
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb
            )

            # Rileva le pose nel frame
            risultato = detector.detect(mp_image)

            if risultato.pose_landmarks:
                # Ordina le persone da sinistra a destra per assegnare gli ID
                persone_con_centro = []
                for landmarks in risultato.pose_landmarks:
                    centro_x, _ = calcola_centro(landmarks)
                    persone_con_centro.append((centro_x, landmarks))

                persone_con_centro.sort(key=lambda p: p[0])

                for pid, (_, landmarks) in enumerate(persone_con_centro, start=1):
                    riga = [contatore_frame, pid]

                    for indice_coco in range(17):
                        indice_mp = MAPPING_MP_A_COCO[indice_coco]
                        lm = landmarks[indice_mp]
                        # Le coordinate di MediaPipe sono già normalizzate (0-1)
                        riga.append(round(lm.x, 4))
                        riga.append(round(lm.y, 4))

                    writer.writerow(riga)

            contatore_frame += 1

    video.release()
    return contatore_frame



def main():
    splits = ["train", "val"]
    categorie = ["fight", "no_fight"]

    # Crea le cartelle di output
    for split in splits:
        for categoria in categorie:
            os.makedirs(os.path.join(CARTELLA_DATASET, split, categoria), exist_ok=True)
            os.makedirs(os.path.join(CARTELLA_OUTPUT, split, categoria), exist_ok=True)

    # Inizializza MediaPipe PoseLandmarker
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODELLO_MP),
        num_poses=MAX_PERSONE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    numero_video_totali = 0

    for split in splits:
        print(f"\n{'='*50}")
        print(f" Split: {split.upper()} — Estrazione con MediaPipe (puro)")
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

                numero_frame = estrazione_video_csv(detector, percorso_video, percorso_csv)

                if numero_frame > 0:
                    print(f"Successo: {numero_frame} frame | {time.time()-tempo_inizio:.1f}s")
                else:
                    print(f"Errore | {time.time()-tempo_inizio:.1f}s")

                numero_video_totali += 1

    detector.close()

    print(f"\n{'='*50}")
    print(f" Video processati: {numero_video_totali}")
    print(f" Output in: {CARTELLA_OUTPUT}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
