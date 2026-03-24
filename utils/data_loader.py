"""
utils/data_loader.py

TAF : Préparation des données
Mission : Chargez les données et normalisez les pixels (mise à l’échelle
entre 0 et 1). Intégrez des couches de Data Augmentation natives à TensorFlow (ex :
tf.keras.layers.RandomFlip, RandomRotation) directement dans votre pipeline ou au
début de votre modèle.

"""
#Importation des différentes dépendances logicielles
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
    tf.keras.layers.RandomZoom(height_factor=0.1),   # zoom léger
], name="data_augmentation")

# Constantes du modèle
BATCH_SIZE = 50
AUTOTUNE   = tf.data.AUTOTUNE

def make_datasets(x_train, y_train, x_test, y_test, batch_size=BATCH_SIZE):
    
    # Dataset d'entraînement (avec augmentation)
    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)) #Decoupage des images en fonction de leur label
        .shuffle(buffer_size=10_000, seed=42) #Le mélange des données
        .batch(batch_size) #Envoi les images en lot de batch_size =50 pour rendre l'entraîment plus rapide
        .map(lambda x, y: (augmentation_layers(x, training=True), y),
             num_parallel_calls=AUTOTUNE) #augmentation sur train uniquement
        .prefetch(AUTOTUNE) #Preparation du prochain lot(batch) pendant l'entraîment du lot actuel
    )

    # Dataset de test (sans augmentation)
    test_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    print(f"[DATA] Batch size : {batch_size} | Augmentation : activée sur train")

    return train_ds, test_ds

