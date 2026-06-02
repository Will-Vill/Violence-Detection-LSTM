import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

# --- CONFIGURAZIONE ---
# Inserisci qui il percorso di un video di rissa dove le persone si muovono molto
PERCORSO_VIDEO = "datasets/train/fight/RealLife_V_1.mp4" 
MODELLO_MP = "pose_landmarker_full.task" # Assicurati di aver scaricato questo file
MAX_PERSONE = 5

# Hardcodiamo le connessioni (le ossa) dello scheletro di MediaPipe.
# In questo modo bypassiamo i bug del modulo 'solutions' e rendiamo il codice indipendente.
CONNESSIONI_MEDIAPIPE = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32)
]

def calcola_centro_pixel(landmarks, w, h):
    """
    Calcola il centro tra le due anche per ordinare le persone da sinistra a destra.
    Le coordinate sono normalizzate (0-1), quindi le moltiplichiamo per width e height.
    """
    anca_sx = landmarks[23]
    anca_dx = landmarks[24]
    
    centro_x_norm = (anca_sx.x + anca_dx.x) / 2
    centro_y_norm = (anca_sx.y + anca_dx.y) / 2
    
    # Converti in coordinate pixel
    return int(centro_x_norm * w), int(centro_y_norm * h)

def main():
    # Inizializza il PoseLandmarker di MediaPipe Tasks (Multi-Person API nuova)
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODELLO_MP),
        num_poses=MAX_PERSONE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(PERCORSO_VIDEO)
    
    if not cap.isOpened():
        print(f"ERRORE: Impossibile aprire il video {PERCORSO_VIDEO}")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("Premi 'q' per chiudere il video.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe richiede immagini in formato RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Elaborazione (Estrazione Posa)
        risultato = detector.detect(mp_image)

        if risultato.pose_landmarks:
            # Lista per salvare le persone e il loro centro per ordinarle
            persone_con_centro = []

            for landmarks in risultato.pose_landmarks:
                centro_x, centro_y = calcola_centro_pixel(landmarks, frame_w, frame_h)
                persone_con_centro.append((centro_x, centro_y, landmarks))

            # ORDINAMENTO DA SINISTRA A DESTRA (Il trucco dell'ID temporaneo)
            persone_con_centro.sort(key=lambda p: p[0])

            # DISEGNO SUL FRAME CUSTOM CON OPENCV
            for pid, (cx, cy, landmarks) in enumerate(persone_con_centro, start=1):
                
                # 1. Convertiamo tutti i 33 punti in coordinate pixel (x, y)
                punti_pixel = []
                for lm in landmarks:
                    px = int(lm.x * frame_w)
                    py = int(lm.y * frame_h)
                    punti_pixel.append((px, py))

                # 2. Disegniamo le linee (le ossa dello scheletro) usando la nostra lista fissa
                for connessione in CONNESSIONI_MEDIAPIPE:
                    punto_inizio = punti_pixel[connessione[0]]
                    punto_fine = punti_pixel[connessione[1]]
                    cv2.line(frame, punto_inizio, punto_fine, (0, 255, 0), 2) # Linea verde

                # 3. Disegniamo i punti (le giunture)
                for punto in punti_pixel:
                    cv2.circle(frame, punto, 3, (0, 0, 255), -1) # Pallino rosso

                # 4. Disegna il "Centro del bacino" usato per il calcolo ID (Pallino giallo)
                cv2.circle(frame, (cx, cy), 8, (0, 255, 255), -1)

                # 5. Trova la posizione del naso (landmark 0) per scriverci l'ID sopra la testa
                naso = punti_pixel[0]
                
                # Disegna il testo dell'ID sopra la testa in giallo
                cv2.putText(frame, f"ID: {pid}", (naso[0] - 20, max(20, naso[1] - 20)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

        # Mostra il frame a schermo
        cv2.imshow("Test Tracciamento MediaPipe", frame)

        # Ritardo di 30 millisecondi per visualizzarlo a velocità normale
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    main()