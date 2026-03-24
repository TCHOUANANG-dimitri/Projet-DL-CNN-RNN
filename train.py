from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from models.cnn_model import CustomCNN 
from utils.cnn_data_loader import load_and_preprocess,make_datasets


(train_ds,test_ds)=make_datasets(load_and_preprocess())

early_stopping= EarlyStopping(
    monitor='val_loss',
    patience=10,
    min_delta=0.001,
    mode='auto',
    restore_best_weights=True,
    start_from_epoch=5

)
checkpoint=ModelCheckpoint(
    filepath='Best_model.keras',
    monitor='val_loss',
    save_best_only=True
)


model=CustomCNN(num_classes=10)
model.compile(
    optimizer='adam',
    loss="SparseCategoricalCrossentropy",
    metrics=['accuracy']
)

history=model.fit(train_ds[0],train_ds[1],batch_size=100,epochs=20,validation_split=0.15,callbacks=[early_stopping,checkpoint])







