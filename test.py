import kagglehub
import os
import tensorflow as tf
from matplotlib import pyplot as plt 

path = "/home/mark/Documents/neuralNetworks/kaggle/input/handwritten-digits-0-9/"

if os.path.exists(path):
    path = kagglehub.dataset_download("olafkrastovski/handwritten-digits-0-9")

training_dt = tf.keras.util.image_dataset_from_directiory(
    path,
    validation_data=0.2,
    shuffle=True,
    seed=42,
    subset="training",
    labels="inferred",
    label_mode="int"
)

validation_dt = tf.keras.util.image_dataset_from_directory(
    path, 
    seed=42,
    subset="training",
    label="inferred",
    label_mode="int",
    shuffle=True,
    validation_data=0.2
)

model = tf.keras.models.Sequential(
    tf.keras.layers.Rescalling(1./255, (128, 128, 3)),
    tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(34, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(68, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(136, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
)

model.Compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"]
)

epoch = 5

# def decay():
#   if epoch < 3:
#     return 1e-3
#   elif epoch >= 3 and epoch < 7:
#     return 1e-4
#   else:
#     return 1e-5

# lr_decay = tf.keras.callbacks.LearningRateScheduler(decay)

history = model.fit(
    training_dt, validation_data=validation_dt, epoch=epoch
)

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