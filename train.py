
import os

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from models.cnn_model import CustomCNN
from utils.data_loader import load_and_preprocess, make_datasets, BATCH_SIZE
from utils.visualization import plot_learning_curves

(x_train, y_train), (x_test, y_test) = load_and_preprocess()
train_ds, test_ds = make_datasets(x_train, y_train, x_test, y_test)

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


model = CustomCNN(num_classes=10)
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=50 * (50000 // 64)  # 50 époques × steps/époque
)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='SparseCategoricalCrossentropy',
    metrics=['accuracy']
)

if __name__ == '__main__':

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,       # divise le LR par 2
        patience=4,       # attend 4 époques sans amélioration
        min_lr=1e-6,
        verbose=1
   )

    history = model.fit(
        train_ds,
        epochs=50,
        validation_data=test_ds,
        callbacks=[early_stopping, checkpoint, reduce_lr]
    )

    os.makedirs("figures", exist_ok=True)
    plot_learning_curves(history.history, save_path="figures/learning_curves.png")
