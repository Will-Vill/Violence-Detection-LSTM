import os
# Fix per Wayland su Linux (stesso di MediaPipe)
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
from ultralytics import YOLO

# --- CONFIGURAZIONE ---
# Usa lo stesso identico video che hai usato per MediaPipe
PERCORSO_VIDEO = "datasets/train/fight/RealLife_V_1.mp4" 

# Usa il modello YOLO-Pose che hai addestrato/scaricato
# Assicurati che il nome del file sia corretto (quello che hai nella cartella)
MODELLO_YOLO = "yolo26n-pose.pt" 

def main():
    print(f"Caricamento modello {MODELLO_YOLO}...")
    
    # Inizializza il modello YOLO
    modello = YOLO(MODELLO_YOLO)

    cap = cv2.VideoCapture(PERCORSO_VIDEO)
    
    if not cap.isOpened():
        print(f"ERRORE: Impossibile aprire il video {PERCORSO_VIDEO}")
        return

    print("Premi 'q' per chiudere il video.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # LA MAGIA DI YOLO È QUI:
        # persist=True attiva il Tracker interno (ByteTrack/BoT-SORT)
        # tracker="botsort.yaml" (opzionale, ma di default usa algoritmi avanzatissimi)
        risultati = modello.track(frame, persist=True, verbose=False)

        # Prendi il risultato del primo (e unico) frame processato
        r = risultati[0]

        # YOLO ha una funzione nativa '.plot()' che disegna automaticamente:
        # 1. Il Bounding Box (rettangolo)
        # 2. Lo scheletro (17 keypoints)
        # 3. L'ID del tracker sopra il rettangolo (es. '1', '2')
        frame_disegnato = r.plot()

        # Mostra il frame a schermo
        cv2.imshow("Test Tracciamento YOLO-Pose (Detection + Tracking)", frame_disegnato)

        # Ritardo di 30 millisecondi per visualizzarlo a velocità normale
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()