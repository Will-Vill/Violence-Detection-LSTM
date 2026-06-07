import os
import shutil
import random

random.seed(42)

CARTELLA_SORGENTE = "/home/williamvil/Scaricati/archive/data"  # cartella unica con tutti i video
CARTELLA_DESTINAZIONE = "datasets"           # stessa usata dagli altri dataset

def splitta_hockey(cartella_origine, percentuale_train=0.8):
    
    tutti_video = [f for f in os.listdir(cartella_origine) 
                   if f.endswith((".avi", ".mp4"))]
    
    # Separa per classe dal nome file
    fight_video    = [f for f in tutti_video if f.startswith("fi")]
    nofight_video  = [f for f in tutti_video if f.startswith("no")]
    
    print(f"Trovati: {len(fight_video)} fight, {len(nofight_video)} no-fight")
    
    for lista, categoria in [(fight_video, "fight"), (nofight_video, "no_fight")]:
        random.shuffle(lista)
        taglio = int(len(lista) * percentuale_train)
        train = lista[:taglio]
        val   = lista[taglio:]
        
        for split_nome, video_lista in [("train", train), ("val", val)]:
            dst_dir = os.path.join(CARTELLA_DESTINAZIONE, split_nome, categoria)
            os.makedirs(dst_dir, exist_ok=True)
            for video in video_lista:
                src = os.path.join(cartella_origine, video)
                dst = os.path.join(dst_dir, "Hockey_" + video)
                shutil.copy2(src, dst)
        
        print(f"{categoria}: {taglio} train, {len(lista)-taglio} val")

if __name__ == "__main__":
    splitta_hockey(CARTELLA_SORGENTE)