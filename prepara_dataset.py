import os
import shutil
import random

# Fissa il seed per essere sicuro che ogni volta che lo lanci, lo split sia identico
random.seed(42)

# --- INSERISCI QUI I PERCORSI DEL NUOVO DATASET SCARICATO ---
# Esempio: dove si trovano attualmente le cartelle "Violence" e "NonViolence"
CARTELLA_SORGENTE_VIOLENCE = "/home/williamvil/Scaricati/archive/Real Life Violence Dataset//Violence"
CARTELLA_SORGENTE_NONVIOLENCE = "/home/williamvil/Scaricati/archive/Real Life Violence Dataset//NonViolence"

# --- LA TUA CARTELLA DI DESTINAZIONE (quella letta dal tuo script 1) ---
CARTELLA_DESTINAZIONE = "datasets"

def splitta_e_sposta(cartella_origine, categoria_destinazione, percentuale_train=0.8):
    """
    Prende i video, li mischia, e li divide 80% train / 20% val.
    categoria_destinazione deve essere "fight" o "no_fight".
    """
    # 1. Trova tutti i video
    lista_video = [f for f in os.listdir(cartella_origine) if f.endswith((".mp4", ".avi", ".mov", ".mkv"))]
    
    if not lista_video:
        print(f"ATTENZIONE: Nessun video trovato in {cartella_origine}")
        return

    # 2. Mischiali a caso (fondamentale per non avere video tutti uguali di fila)
    random.shuffle(lista_video)

    # 3. Calcola il punto di taglio per l'80%
    punto_taglio = int(len(lista_video) * percentuale_train)
    video_train = lista_video[:punto_taglio]
    video_val = lista_video[punto_taglio:]

    print(f"Trovati {len(lista_video)} video in {cartella_origine}.")
    print(f" -> {len(video_train)} andranno in TRAIN")
    print(f" -> {len(video_val)} andranno in VAL")

    # 4. Sposta (copia) i file nelle cartelle finali
    # Creiamo i percorsi di destinazione
    dir_train = os.path.join(CARTELLA_DESTINAZIONE, "train", categoria_destinazione)
    dir_val = os.path.join(CARTELLA_DESTINAZIONE, "val", categoria_destinazione)

    os.makedirs(dir_train, exist_ok=True)
    os.makedirs(dir_val, exist_ok=True)

    print("Copia in corso (Train)...")
    for video in video_train:
        src = os.path.join(cartella_origine, video)
        # Uniamo un prefisso al nome per non sovrascrivere i video di RWF-2000 se hanno lo stesso nome (es. video_1.mp4)
        dst = os.path.join(dir_train, "RealLife_" + video)
        shutil.copy2(src, dst)

    print("Copia in corso (Val)...")
    for video in video_val:
        src = os.path.join(cartella_origine, video)
        dst = os.path.join(dir_val, "RealLife_" + video)
        shutil.copy2(src, dst)
        
    print(f"Completato lo split per la categoria: {categoria_destinazione}\n")

if __name__ == "__main__":
    print("Inizio preparazione del dataset Real Life Violence...")
    
    # Processa i video violenti -> "fight"
    splitta_e_sposta(CARTELLA_SORGENTE_VIOLENCE, "fight")
    
    # Processa i video normali -> "no_fight"
    splitta_e_sposta(CARTELLA_SORGENTE_NONVIOLENCE, "no_fight")
    
    print("Tutti i video sono stati splittati con successo nella cartella 'datasets'!")