"""
utils/data_loader.py

TAF : Préparation des données
Mission : Chargez les données et normalisez les pixels (mise à l’échelle
entre 0 et 1). Intégrez des couches de Data Augmentation natives à TensorFlow (ex :
tf.keras.layers.RandomFlip, RandomRotation) directement dans votre pipeline ou au
début de votre modèle.

"""
#Importation des différentes dépendances logicielles
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Chargement des données de CIFAR-10
def load_and_preprocess():
    #seperation en train et en test
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalisation : diviser par 255 pour avoir des valeurs entre 0 et 1
    x_train = x_train.astype("float32") / 255.0 
    x_test  = x_test.astype("float32")  / 255.0

    print(f"[Forme d'entrée] Train : {x_train.shape} | Test : {x_test.shape}")
    print(f"[Normalisation] Plage des pixels train — min: {x_train.min():.1f} | max: {x_train.max():.1f}")

    return (x_train, y_train), (x_test, y_test)


# Noms des 10 classes du dataset CIFAR-10
CLASS_NAMES = [
    "avion", "voiture", "oiseau", "chat", "cerf",
    "chien", "grenouille", "cheval", "bateau", "camion"
]

# Definition des Couches d'augmentation
augmentation_layers = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),        # miroir horizontal
    tf.keras.layers.RandomRotation(factor=0.1),      # rotation ±10%
    tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),# translation légère
    tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1),   # zoom léger
    tf.keras.layers.RandomContrast(factor=0.1),      # contraste variable
], name="data_augmentation")

# Constantes du modèle
BATCH_SIZE = 64
AUTOTUNE   = tf.data.AUTOTUNE

def make_datasets(x_train, y_train, x_test, y_test, batch_size=BATCH_SIZE, val_split=0.1):
    # Séparation train / validation / test
    n_samples = len(x_train)
    val_size = int(n_samples * val_split)
    indices = np.random.default_rng(42).permutation(n_samples)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    x_train_split = x_train[train_idx]
    y_train_split = y_train[train_idx]
    x_val = x_train[val_idx]
    y_val = y_train[val_idx]

    # Dataset d'entraînement (avec augmentation)
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train_split, y_train_split))
        .shuffle(buffer_size=10_000, seed=42)
        .batch(batch_size)
        .map(lambda x, y: (augmentation_layers(x, training=True), y),
             num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    # Dataset de validation (aucune augmentation)
    val_ds = (
        tf.data.Dataset.from_tensor_slices((x_val, y_val))
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    # Dataset de test (réservé à l'évaluation finale)
    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    print(f"[DATA] Batch size : {batch_size} | Augmentation : activée sur train | Validation : {val_split*100:.0f}%")

    return train_ds, val_ds, test_ds

