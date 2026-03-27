
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from models.cnn_model import CustomCNN
from utils.data_loader import load_and_preprocess, make_datasets, BATCH_SIZE

(x_train, y_train), (x_test, y_test) = load_and_preprocess()
train_ds, test_ds = make_datasets(x_train, y_train, x_test, y_test)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
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

    history = model.fit(
        train_ds,
        epochs=20,
        validation_data=test_ds,
        callbacks=[early_stopping, checkpoint]
    )
