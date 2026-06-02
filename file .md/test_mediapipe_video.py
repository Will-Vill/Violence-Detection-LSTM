import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

# --- CONFIG ---
PERCORSO_VIDEO = "/Users/williamvil/Desktop/Tesi/AI_Tesi/file .md/RealLife_NV_1.mp4"
MODELLO_MP = "pose_landmarker.task"
MAX_PERSONE = 5

CONNESSIONI_MEDIAPIPE = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
    (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
    (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (28, 32)
]

def calcola_centro_pixel(landmarks, w, h):
    anca_sx = landmarks[23]
    anca_dx = landmarks[24]

    centro_x = (anca_sx.x + anca_dx.x) / 2
    centro_y = (anca_sx.y + anca_dx.y) / 2

    return int(centro_x * w), int(centro_y * h)


def main():

    # --- MediaPipe init ---
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODELLO_MP),
        num_poses=MAX_PERSONE,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(PERCORSO_VIDEO)

    if not cap.isOpened():
        print("Errore apertura video")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    paused = False
    current_frame = None

    print("p = pausa | s = salva frame | q = esci")

    while True:

        key = cv2.waitKey(1) & 0xFF

        # --- PAUSA ---
        if key == ord('p'):
            paused = not paused

        # --- LETTURA FRAME SOLO SE NON IN PAUSA ---
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            current_frame = frame.copy()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb
            )

            risultato = detector.detect(mp_image)

            if risultato.pose_landmarks:

                persone = []

                for landmarks in risultato.pose_landmarks:
                    cx, cy = calcola_centro_pixel(landmarks, frame_w, frame_h)
                    persone.append((cx, cy, landmarks))

                persone.sort(key=lambda p: p[0])

                for pid, (cx, cy, landmarks) in enumerate(persone, start=1):

                    punti = []
                    for lm in landmarks:
                        x = int(lm.x * frame_w)
                        y = int(lm.y * frame_h)
                        punti.append((x, y))

                    # ossa
                    for a, b in CONNESSIONI_MEDIAPIPE:
                        cv2.line(frame, punti[a], punti[b], (0, 255, 0), 2)

                    # punti
                    for p in punti:
                        cv2.circle(frame, p, 3, (0, 0, 255), -1)

                    # centro bacino
                    cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1)

                    # ID
                    nose = punti[0]
                    cv2.putText(
                        frame,
                        f"ID {pid}",
                        (nose[0], nose[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2
                    )

        # --- MOSTRA VIDEO ---
        if current_frame is not None:
            cv2.imshow("MediaPipe Pose", frame)

        # --- SALVA FRAME ---
        if key == ord('s') and current_frame is not None:
            cv2.imwrite("frame_originale.png", current_frame)
            cv2.imwrite("frame_mediapipe.png", frame)  # <--- IL FIX È QUI
            print("Frame salvato!")

        # --- EXIT ---
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == "__main__":
    main()