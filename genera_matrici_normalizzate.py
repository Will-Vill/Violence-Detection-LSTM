"""
Script per generare le matrici di confusione NORMALIZZATE
Stile identico a 2_addestra_lstm_v2.py (cmap='Blues', fontsize=20, fontweight='bold')

Valori presi direttamente dalle matrici di confusione originali della tesi.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("grafici", exist_ok=True)

# ── Dati delle 3 matrici (valori assoluti dalle immagini originali) ────

# 1. Modello Base (79%) — dalla matrice confusion_matrix_79.png
cm_base = np.array([
    [2579, 1306],   # Riga No-Fight: [TN, FP]
    [797,  5204]    # Riga Fight:    [FN, TP]
])

# 2. Modello Ottimizzato (84%) — dalla matrice confusion_matrix_84.png
cm_ottimizzato = np.array([
    [2546, 434],    # Riga No-Fight: [TN, FP]
    [461,  2059]    # Riga Fight:    [FN, TP]
])

# 3. Test Set UCF-Crime (85%) — dalla matrice confusion_matrix_test.png
cm_test = np.array([
    [133, 17],      # Riga No-Fight: [TN, FP]
    [13,  37]       # Riga Fight:    [FN, TP]
])

# ── Funzione di plot (stile identico a 2_addestra_lstm_v2.py) ──────────

def plot_confusion_matrix(cm, titolo, nome_file):
    """
    Genera una matrice di confusione normalizzata per riga (ogni riga somma a 1.0).
    """
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["no_fight", "fight"])
    ax.set_yticklabels(["no_fight", "fight"])
    ax.set_xlabel("Predetto")
    ax.set_ylabel("Reale")
    ax.set_title(titolo)

    for i in range(2):
        for j in range(2):
            valore = f"{cm_norm[i, j]:.2f}"
            colore = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, valore, ha='center', va='center',
                    fontsize=20, fontweight='bold', color=colore)

    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(f"grafici/{nome_file}", dpi=150)
    plt.close()
    print(f"Grafico salvato: grafici/{nome_file}")


# ── Generazione ────────────────────────────────────────────────────────

print("=" * 60)
print(" GENERAZIONE MATRICI DI CONFUSIONE NORMALIZZATE")
print("=" * 60)

plot_confusion_matrix(
    cm_base,
    "Matrice di Confusione Normalizzata — Modello Base",
    "confusion_matrix_79_norm.png"
)

plot_confusion_matrix(
    cm_ottimizzato,
    "Matrice di Confusione Normalizzata — Modello Ottimizzato",
    "confusion_matrix_84_norm.png"
)

plot_confusion_matrix(
    cm_test,
    "Matrice di Confusione Normalizzata — Test Set",
    "confusion_matrix_test_norm.png"
)

print(f"\n{'=' * 60}")
print(" FATTO! 3 matrici normalizzate generate in grafici/")
print(f"{'=' * 60}")

# ── Verifica ───────────────────────────────────────────────────────────

print("\n--- Verifica valori normalizzati ---\n")
for nome, cm in [("Base (79%)", cm_base), 
                  ("Ottimizzato (84%)", cm_ottimizzato), 
                  ("Test Set (85%)", cm_test)]:
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    print(f"{nome}:")
    print(f"  Recall No-Fight: {cm_norm[0, 0]:.2%}")
    print(f"  Recall Fight:    {cm_norm[1, 1]:.2%}")
    print()
