import tensorflow as tf

# 1. Dataset setup
data = r"C:\Users\yoges\Downloads\archive\Training"


train = tf.keras.utils.image_dataset_from_directory(data, subset="training", validation_split=0.2, seed=1, image_size=(224,224))
val = tf.keras.utils.image_dataset_from_directory(data, subset="validation", validation_split=0.2, seed=1, image_size=(224,224))

# 2. Model structure
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'), tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'), tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'), tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(train.class_names), activation='softmax')
])

# 3. Training
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
checkpoint = tf.keras.callbacks.ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True)

h = model.fit(train, validation_data=val, epochs=10, callbacks=[checkpoint])

print("Accuracy:", h.history['accuracy'][-1])
