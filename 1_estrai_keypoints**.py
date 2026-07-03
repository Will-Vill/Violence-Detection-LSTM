import os
import csv
import glob
import cv2
import time
from ultralytics import YOLO

# CONFIGURAZIONE
MODELLO = "yolo26n-pose.pt"
CARTELLA_DATASET = "datasets/RWF2000_RealLifeViolence"
CARTELLA_OUTPUT = "output"

NOMI_KP = [
    "naso", "occhio_sx", "occhio_dx", "orecchio_sx", "orecchio_dx",
    "spalla_sx", "spalla_dx", "gomito_sx", "gomito_dx", "polso_sx", "polso_dx",
    "anca_sx", "anca_dx", "ginocchio_sx", "ginocchio_dx", "caviglia_sx", "caviglia_dx"
]

def estrazione_video_csv(modello, percorso_video, percorso_csv):
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

            risultati = modello.track(frame, persist=True, verbose=False, device=0)
            r = risultati[0]

            if r.keypoints is not None and r.boxes is not None and r.boxes.id is not None:
                kp_persone = r.keypoints.xy.cpu().numpy()
                id_persone = r.boxes.id.cpu().numpy().astype(int)

                for i, pid in enumerate(id_persone):
                    riga = [contatore_frame, pid]
                    for indice_kp in range(17):
                        riga.append(round(kp_persone[i][indice_kp][0] / frame_w, 4)) # x normalizzata
                        riga.append(round(kp_persone[i][indice_kp][1] / frame_h, 4)) # y normalizzata
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

    modello = YOLO(MODELLO)
    numero_video_totali = 0

    for split in splits:
        print(f"\n{'='*50}")
        print(f" Split: {split.upper()}")
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

                numero_frame = estrazione_video_csv(modello, percorso_video, percorso_csv)

                if numero_frame > 0:
                    print(f"Successo: {numero_frame} frame | {time.time()-tempo_inizio:.1f}s")
                else:
                    print(f"Errore | {time.time()-tempo_inizio:.1f}s")

                numero_video_totali += 1

    print(f"\n{'='*50}")
    print(f" Video processati: {numero_video_totali}")
    print(f" Output in: {CARTELLA_OUTPUT}/")


if __name__ == "__main__":
    main()