"""
evaluate.py
───────────
Charge le meilleur modèle sauvegardé (Best_model.keras) et produit
toutes les visualisations + métriques pour le rapport.

Usage :
    python evaluate.py
    python evaluate.py --model Best_model.keras
"""

import argparse
import os
import numpy as np
import tensorflow as tf

from models.cnn_model import CustomCNN
from utils.data_loader import load_and_preprocess, make_datasets
from utils.visualization import (
    load_history,
    plot_confusion_matrix,
    plot_learning_curves,
    print_classification_report,
    plot_per_class_metrics,
    plot_predictions,
    plot_misclassified,
)

OUTPUT_DIR = "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate(model_path: str):

    # ── 1. Données ────────────────────────────────────────────────────────────
    print("\n[1/5] Chargement des données CIFAR-10...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess()
    _, _, test_ds = make_datasets(x_train, y_train, x_test, y_test)
    # test_ds est réservé à l'évaluation finale ; validation est utilisée uniquement pendant l'entraînement.

    # y_test shape (10000, 1) → on aplatit en (10000,)
    y_true = y_test.flatten().astype(int)

    # ── 2. Modèle ─────────────────────────────────────────────────────────────
    print(f"[2/5] Chargement du modèle : {model_path}")
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"CustomCNN": CustomCNN}
    )
    model.summary() #resumer des choix
    # Sauvegarder le summary dans un fichier texte
    with open(os.path.join(OUTPUT_DIR, "model_summary.txt"), "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    print("[✓] Architecture sauvegardée → figures/model_summary.txt")

    history_path = os.path.join(OUTPUT_DIR, "training_history.json")
    if os.path.exists(history_path):
        history = load_history(history_path)
        plot_learning_curves(
            history,
            save_path=os.path.join(OUTPUT_DIR, "learning_curves.png")
        )
    else:
        print("[i] Historique d'entraînement introuvable. Courbes d'apprentissage non tracées.")

    # ── 3. Évaluation globale ─────────────────────────────────────────────────
    print("\n[3/5] Évaluation sur le jeu de test...")
    loss, accuracy = model.evaluate(test_ds, verbose=1)

    # ── 4. Prédictions ────────────────────────────────────────────────────────
    print("\n[4/5] Génération des prédictions...")
    y_pred_proba = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)

    # ── 5. Visualisations ─────────────────────────────────────────────────────
    print("\n[5/5] Génération des visualisations...\n")

    # Rapport textuel precision / recall / F1
    print_classification_report(y_true, y_pred)

    # Matrice de confusion
    plot_confusion_matrix(
        y_true, y_pred,
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    )

    # Barplot precision / recall / F1 par classe
    plot_per_class_metrics(
        y_true, y_pred,
        save_path=os.path.join(OUTPUT_DIR, "per_class_metrics.png")
    )

    # 25 exemples aléatoires du jeu de test
    rng = np.random.default_rng(42)
    idx = rng.choice(len(x_test), size=25, replace=False)
    plot_predictions(
        x_test[idx], y_true[idx], y_pred[idx],
        n=25,
        save_path=os.path.join(OUTPUT_DIR, "predictions_samples.png")
    )

    # Exemples mal classifiés
    plot_misclassified(
        x_test, y_true, y_pred,
        n=20,
        save_path=os.path.join(OUTPUT_DIR, "misclassified.png")
    )

    # ── Résumé ────────────────────────────────────────────────────────────────
    n_errors = int((y_true != y_pred).sum())
    print("\n" + "=" * 55)
    print("  RÉSUMÉ FINAL")
    print("=" * 55)
    print(f"  Modèle          : {model_path}")
    print(f"  Test Loss       : {loss:.4f}")
    print(f"  Test Accuracy   : {accuracy * 100:.2f} %")
    print(f"  Nb échantillons : {len(y_true)}")
    print(f"  Nb erreurs      : {n_errors}  ({n_errors/len(y_true)*100:.1f} %)")
    print("=" * 55)
    print(f"\n  Figures sauvegardées dans → ./{OUTPUT_DIR}/")
    print("    • confusion_matrix.png")
    print("    • per_class_metrics.png")
    print("    • predictions_samples.png")
    print("    • misclassified.png\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation CNN CIFAR-10")
    parser.add_argument(
        "--model",
        type=str,
        default="Best_model.keras",
        help="Chemin vers le modèle sauvegardé (.keras ou .h5)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Modèle introuvable : '{args.model}'\n"
            "Lance d'abord train.py ou vérifie le chemin."
        )

    evaluate(args.model)