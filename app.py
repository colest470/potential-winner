import kagglehub
import os
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

path = "./kaggle/input/handwritten-digits-0-9/"
saved_model_path = "./european_digits_model.keras"

def laod():
    if os.path.exists("./european_digits_model.keras"):
        model = tf.keras.models.load_model("./european_digits_model.keras")
        print(f"Model loaded from {saved_model_path}")

        model.summary()

        return model

    if os.path.exists(path) == False:
        path = kagglehub.dataset_download("olafkrastovski/handwritten-digits-0-9")
        train_model()

def train_model():
    training_dt = tf.keras.util.image_dataset_from_directory(
        subset="training",
        validation_split=0.2,
        shuffle=True,
        label="inferred",
        label_mode="int",
        seed=42
    )

    validation_dt = tf.keras.util.image_dataset_from_directory(
        subset="validation",
        validation_split=0.2,
        label="inferred",
        label_mode="int",
        shuffle=True,
        seed=42
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.Rescalling(1./255, (128, 128, 3)),
        tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2d(),
        tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),

        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    epoch = 5

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learnin_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), # because of softmax in last layer
        metrics=["accuracy"]
    )

    def decay():
        if epoch >= 3:
            return 1e-3
        elif epoch >= 2:
            return 1e-4
        else:
            return 1e-5
        
    lr_decay = tf.keras.callbacks.LearningRateScheduler(decay)

    history = model.fit(training_dt, validation_dt, epoch, callback=[lr_decay])

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epoch)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def load_and_preprocess_image(image_path, size=(128, 128)):
  img = tf.keras.utils.load_img(image_path, target_size=size)
  img_arr = tf.keras.utils.img_to_array(img)
  img_batch = tf.expand_dims(img_arr, 0)

  return img_batch, img_arr

img_batch, img_display = load_and_preprocess_image("/content/seven1.jpeg", size=(128, 128))

if __name__ == "__main__":
    saved_model = load()
    predictions = saved_model.predict(img_batch, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])

    print(predicted_class_idx)
    print(f"Predicted Label: {predicted_class_idx}")

    plt.figure(figsize=(4, 4))
    plt.imshow(img_display.astype("uint8"))
    color = 'black'
    plt.title(f"Predicted: {predicted_class_idx}", color=color)
    plt.axis("off")
    plt.show()