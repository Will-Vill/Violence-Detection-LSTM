import cv2
from ultralytics import YOLO

PERCORSO_VIDEO = "/Users/williamvil/Desktop/Tesi/AI_Tesi/file .md/RealLife_NV_1.mp4"
MODELLO_YOLO = "yolo26n-pose.pt"

def main():
    modello = YOLO(MODELLO_YOLO)

    cap = cv2.VideoCapture(PERCORSO_VIDEO)

    if not cap.isOpened():
        print("Errore apertura video")
        return

    paused = False
    saved = False

    print("p = pausa | s = salva frame | q = esci")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            risultati = modello.track(frame, persist=True, verbose=False)
            r = risultati[0]
            frame_disegnato = r.plot()

            current_frame = frame.copy()
            current_drawn = frame_disegnato.copy()

        cv2.imshow("YOLO-Pose", current_drawn)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('p'):
            paused = not paused  # freeze/unfreeze

        elif key == ord('s'):
            cv2.imwrite("frame_originale.png", current_frame)
            cv2.imwrite("frame_yolo.png", current_drawn)
            print("Frame salvati!")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()