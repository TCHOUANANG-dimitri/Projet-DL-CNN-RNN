
import os
import json

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from models.cnn_model import CustomCNN
from utils.data_loader import load_and_preprocess, make_datasets, BATCH_SIZE
from utils.visualization import plot_learning_curves, save_history

(x_train, y_train), (x_test, y_test) = load_and_preprocess()
# Séparation explicite train / validation / test
train_ds, val_ds, test_ds = make_datasets(x_train, y_train, x_test, y_test)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=7,
    min_delta=0.001,
    mode='auto',
    restore_best_weights=True,
    start_from_epoch=5
)
checkpoint = ModelCheckpoint(
    filepath='Best_model.keras',
    monitor='val_loss',
    save_best_only=True
)


class SaveHistoryCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for key, value in logs.items():
            self.history.setdefault(key, []).append(float(value))
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        save_history(self.history, save_path=self.save_path)


model = CustomCNN(num_classes=10)
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=50 * (50000 // BATCH_SIZE)  
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='SparseCategoricalCrossentropy',
    metrics=['accuracy']
)

if __name__ == '__main__':
    figures_dir = "figures"
    os.makedirs(figures_dir, exist_ok=True)
    history_save_path = os.path.join(figures_dir, "training_history.json")
    curve_save_path = os.path.join(figures_dir, "learning_curves.png")

    save_history_cb = SaveHistoryCallback(save_path=history_save_path)
    callbacks = [early_stopping, checkpoint, save_history_cb]

    history = None
    try:
        history = model.fit(
            train_ds,
            epochs=45,
            validation_data=val_ds,
            callbacks=callbacks
        )
    except KeyboardInterrupt:
        print("\n[i] Entraînement interrompu manuellement.")
    finally:
        final_history = history.history if history is not None else save_history_cb.history
        if final_history:
            save_history(final_history, save_path=history_save_path)
            plot_learning_curves(final_history, save_path=curve_save_path, show=False)
            print(f"[✓] Historique sauvegardé dans {figures_dir}/")
        else:
            print("[i] Aucun historique disponible à sauvegarder.")
