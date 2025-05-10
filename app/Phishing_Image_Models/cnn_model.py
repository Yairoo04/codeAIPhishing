import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, applications
import matplotlib.pyplot as plt
from data_loader import load_dataset

DATA_DIR = '../dataset_Image'
MODEL_OUT = '../models/cnn_phishing_image.keras'
IMG_SIZE = (128, 128)
EPOCHS = 30

def build_cnn(input_shape=(*IMG_SIZE, 3), learning_rate=1e-4):
    base_model = applications.EfficientNetB0(
        weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def plot_history(history, save_path='training_history.png'):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def main():
    train_ds, val_ds = load_dataset(DATA_DIR)

    model = build_cnn()
    model.summary()

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early_stop, reduce_lr]
    )

    plot_history(history)

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save(MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")

if __name__ == '__main__':
    main()
