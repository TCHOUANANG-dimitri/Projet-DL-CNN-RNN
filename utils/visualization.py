"""
utils/visualization.py
Fonctions de visualisation pour le projet CNN CIFAR-10.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

# Classes en français — cohérent avec data_loader.py
CIFAR10_CLASSES = [
    "avion", "voiture", "oiseau", "chat", "cerf",
    "chien", "grenouille", "cheval", "bateau", "camion"
]


# ─────────────────────────────────────────────
# 1. COURBES D'APPRENTISSAGE
# À appeler dans train.py après model.fit()
# ─────────────────────────────────────────────
def plot_learning_curves(history, save_path=None):
    """
    Affiche les courbes Loss et Accuracy (train vs validation).
    Paramètre : history.history (dict retourné par model.fit)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Courbes d'apprentissage", fontsize=15, fontweight='bold')

    epochs = range(1, len(history['loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['loss'],     'b-o', markersize=4, label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-o', markersize=4, label='Validation')
    axes[0].set_title('Loss (Categorical Cross-Entropy)')
    axes[0].set_xlabel('Époque')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Accuracy
    axes[1].plot(epochs, history['accuracy'],     'b-o', markersize=4, label='Train')
    axes[1].plot(epochs, history['val_accuracy'], 'r-o', markersize=4, label='Validation')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Époque')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[✓] Courbes sauvegardées → {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# 2. MATRICE DE CONFUSION
# ─────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Matrice de confusion normalisée en pourcentage.
    Chaque ligne = classe réelle, chaque colonne = classe prédite.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        cm_norm,
        annot=True, fmt='.1f',
        cmap='Blues',
        xticklabels=CIFAR10_CLASSES,
        yticklabels=CIFAR10_CLASSES,
        linewidths=0.4,
        ax=ax
    )
    ax.set_title('Matrice de Confusion (% par classe réelle)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Classe Prédite', fontsize=12)
    ax.set_ylabel('Classe Réelle', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[✓] Matrice de confusion → {save_path}")
    plt.show()
    return cm


# ─────────────────────────────────────────────
# 3. RAPPORT TEXTUEL (precision / recall / F1)
# ─────────────────────────────────────────────
def print_classification_report(y_true, y_pred):
    """
    Affiche le rapport sklearn complet et le retourne sous forme de chaîne.
    """
    report = classification_report(
        y_true, y_pred,
        target_names=CIFAR10_CLASSES,
        digits=4
    )
    print("\n" + "=" * 62)
    print("         RAPPORT DE CLASSIFICATION")
    print("=" * 62)
    print(report)
    return report


# ─────────────────────────────────────────────
# 4. GRAPHIQUE PRECISION / RECALL / F1 PAR CLASSE
# ─────────────────────────────────────────────
def plot_per_class_metrics(y_true, y_pred, save_path=None):
    """
    Barplot groupé : Precision, Recall, F1-Score pour chaque classe.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(10))
    )

    x = np.arange(len(CIFAR10_CLASSES))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precision, width, label='Precision',  color='steelblue',  alpha=0.85)
    ax.bar(x,         recall,    width, label='Recall',     color='darkorange', alpha=0.85)
    ax.bar(x + width, f1,        width, label='F1-Score',   color='seagreen',   alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=30, ha='right')
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Score')
    ax.set_title('Precision, Recall et F1-Score par classe', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[✓] Métriques par classe → {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# 5. VISUALISATION D'EXEMPLES DE PRÉDICTIONS
# ─────────────────────────────────────────────
def plot_predictions(x_samples, y_true, y_pred, n=25, save_path=None):
    """
    Affiche n images avec la vraie classe et la prédiction.
    Titre vert = correct | rouge = erreur.
    x_samples : numpy array (N, 32, 32, 3), valeurs dans [0, 1]
    """
    n = min(n, len(x_samples))
    cols = 5
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8))
    fig.suptitle('Exemples de prédictions  (vert = correct | rouge = erreur)',
                 fontsize=12, fontweight='bold')
    axes = axes.flatten()

    for i in range(n):
        axes[i].imshow(np.clip(x_samples[i], 0, 1))
        axes[i].axis('off')
        color = 'green' if y_true[i] == y_pred[i] else 'red'
        axes[i].set_title(
            f"R: {CIFAR10_CLASSES[y_true[i]]}\nP: {CIFAR10_CLASSES[y_pred[i]]}",
            fontsize=8, color=color, fontweight='bold'
        )

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[✓] Prédictions → {save_path}")
    plt.show()


# ─────────────────────────────────────────────
# 6. EXEMPLES MAL CLASSIFIÉS
# ─────────────────────────────────────────────
def plot_misclassified(x_test, y_true, y_pred, n=20, save_path=None):
    """
    Affiche les n premières images mal classifiées.
    Permet d'analyser les confusions typiques du modèle.
    """
    errors = np.where(y_true != y_pred)[0]
    print(f"[i] Erreurs totales : {len(errors)} / {len(y_true)} "
          f"({len(errors)/len(y_true)*100:.1f} %)")

    n = min(n, len(errors))
    cols = 5
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.8))
    fig.suptitle(f'Images mal classifiées ({n} exemples)', fontsize=13, fontweight='bold')
    axes = axes.flatten()

    for i, idx in enumerate(errors[:n]):
        axes[i].imshow(np.clip(x_test[idx], 0, 1))
        axes[i].axis('off')
        axes[i].set_title(
            f"✓ {CIFAR10_CLASSES[y_true[idx]]}\n✗ {CIFAR10_CLASSES[y_pred[idx]]}",
            fontsize=8, color='red'
        )

    for j in range(n, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[✓] Images mal classifiées → {save_path}")
    plt.show()