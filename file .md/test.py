import pandas as pd
import matplotlib.pyplot as plt
import time

# I nomi devono corrispondere a quelli del tuo CSV
NOMI_KP = [
    "naso", "occhio_sx", "occhio_dx", "orecchio_sx", "orecchio_dx",
    "spalla_sx", "spalla_dx", "gomito_sx", "gomito_dx", "polso_sx", "polso_dx",
    "anca_sx", "anca_dx", "ginocchio_sx", "ginocchio_dx", "caviglia_sx", "caviglia_dx"
]

# Definizione delle ossa (collegamenti tra i keypoints)
OSSA = [
    (0, 1), (0, 2), (1, 3), (2, 4),                     # Testa
    (5, 6), (5, 11), (6, 12), (11, 12),                 # Busto
    (5, 7), (7, 9),                                     # Braccio SX
    (6, 8), (8, 10),                                    # Braccio DX
    (11, 13), (13, 15),                                 # Gamba SX
    (12, 14), (14, 16)                                  # Gamba DX
]

def visualizza_scheletri(percorso_csv):
    print(f"Caricamento di {percorso_csv}...")
    df = pd.read_csv(percorso_csv)
    
    plt.ion() # Attiva la modalità interattiva
    fig, ax = plt.subplots(figsize=(8, 8))
    
    frames_totali = df['frame'].unique()
    
    for f in frames_totali:
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0) # Invertiamo l'asse Y (lo 0 nei video è in alto)
        ax.set_title(f"Visualizzazione Frame: {f}")
        
        # Prendi tutte le persone in questo frame
        dati_frame = df[df['frame'] == f]
        
        for index, persona in dati_frame.iterrows():
            id_p = int(persona['id_persona'])
            
            # Estrai coordinate x e y per questa persona
            x_coords = [persona[f"{kp}_x"] for kp in NOMI_KP]
            y_coords = [persona[f"{kp}_y"] for kp in NOMI_KP]
            
            # Disegna i pallini (articolazioni)
            ax.scatter(x_coords, y_coords, label=f"ID {id_p}", zorder=5)
            
            # Disegna le linee (ossa)
            for punto1, punto2 in OSSA:
                x1, y1 = x_coords[punto1], y_coords[punto1]
                x2, y2 = x_coords[punto2], y_coords[punto2]
                
                # Disegna l'osso solo se YOLO ha trovato entrambi i punti (non sono 0.0)
                if (x1 != 0.0 and y1 != 0.0) and (x2 != 0.0 and y2 != 0.0):
                    ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.6)
        
        plt.legend(loc='upper right')
        plt.pause(0.03) # Mette in pausa per simulare i ~30 fps del video
        
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    # INSERISCI QUI IL NOME DI UN TUO FILE CSV DA TESTARE
    MIO_FILE = "output/train/no_fight/_fPfNbHM16M_0.csv" 
    visualizza_scheletri(MIO_FILE)