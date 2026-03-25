from tkinter.filedialog import test

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from models.cnn_model import CustomCNN
from utils.data_loader import load_and_preprocess, make_datasets, BATCH_SIZE

(x_train, y_train), (x_test, y_test) = load_and_preprocess()
train_ds, test_ds = make_datasets(x_train, y_train, x_test, y_test)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
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
model.compile(
    optimizer='adam',
    loss='SparseCategoricalCrossentropy',
    metrics=['accuracy']
)

if __name__ == '__main__':
    steps_per_epoch = len(x_train) // BATCH_SIZE
    validation_steps = len(x_test) // BATCH_SIZE0

    history = model.fit(
        train_ds,
        epochs=20,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_ds,
        validation_steps=validation_steps,
        callbacks=[early_stopping, checkpoint]
    )
