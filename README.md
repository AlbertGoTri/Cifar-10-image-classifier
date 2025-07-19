# CIFAR-10 Image Classification with TensorFlow/Keras

This repository contains several progressively improved versions of an image classification model trained on the CIFAR-10 dataset using TensorFlow and Keras. Each version introduces specific architectural or data preprocessing enhancements to improve test accuracy.

## Model Versions Overview

| Version | Description                                      | Test Accuracy |
|--------:|--------------------------------------------------|:-------------:|
|   1     | Simple Dense Model                               |     41%       |
|   2     | Increased Model Depth (more layers)              |     46%       |
|   3     | Data Augmentation                                |     48%       |
|   4     | Convolutional Neural Network (CNN)               |     74%       |
|   5     | Deeper CNN + Learning Rate Scheduler             |     85%       |

---

## Version 1 - Simple Dense Model (41%)

A basic neural network with two dense layers trained on flattened image inputs.

```python
model = Sequential([
    Dense(512, activation='relu', input_shape=(3072,)),
    Dense(10, activation='softmax')
])
```

**Training Plot:**

<img src="https://i.imgur.com/XDQwwhQ.png" alt="Version 1 Plot" width="400"/>

---

## Version 2 - Increased Model Depth (46%)

This version increases the network depth by adding more dense layers and dropout to help prevent overfitting.

```python
model = Sequential([
    Dense(1024, activation='relu', name="L1", input_shape=(3072,)),
    Dropout(0.3),
    Dense(512, activation='relu', name="L2"),
    Dropout(0.3),
    Dense(256, activation='relu', name="L3"),
    Dropout(0.2),
    Dense(10, activation='softmax')
])
```

**Training Plot:**

<img src="https://i.imgur.com/xMvszG4.png" alt="Version 2 Plot" width="400"/>

---

## Version 3 - Data Augmentation (48%)

Introduces data augmentation to improve generalization. A custom generator is used to flatten the augmented images before feeding them into the dense model.

```python
datagen = ImageDataGenerator(
    rotation_range=15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1
)

def generator_with_flatten(datagen, X, y, batch_size):
    gen = datagen.flow(X, y, batch_size=batch_size)
    while True:
        X_batch, y_batch = next(gen)
        X_batch = X_batch.reshape(X_batch.shape[0], -1)
        yield X_batch, y_batch
```

**Training Plot:**

<img src="https://i.imgur.com/QqEQxJS.png" alt="Version 3 Plot" width="400"/>

---

## Version 4 - Convolutional Neural Network (CNN) (74%)

Switches to a CNN architecture to exploit the spatial structure of image data. The model includes convolutional, pooling, batch normalization, and dropout layers.

```python
model = Sequential([
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(32,32,3)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(64, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(128, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

**Training Plot:**

<img src="https://i.imgur.com/EKgZUwo.png" alt="Version 4 Plot" width="400"/>

---

## Version 5 - Deeper CNN + Learning Rate Scheduler (85%)

This version significantly improves performance with a deeper CNN and a `ReduceLROnPlateau` learning rate schedule. It uses two convolutional layers per block and more filters, followed by dropout and batch normalization for regularization.

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
```

Learning rate schedule:

```python
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

history = model.fit(datagen.flow(X_train, y_train, batch_size=64),
                    validation_data=(X_test, y_test),
                    epochs=50, 
                    callbacks=[lr_schedule])
```

**Training Plot:**

<img src="https://i.imgur.com/QPoFto7.png" alt="Version 5 Plot" width="400"/>

---

## Conclusion

Each version builds upon the previous one, introducing improvements that lead to better generalization and higher test accuracy. Version 5, featuring a deeper CNN and learning rate scheduler, achieves the highest performance with 85% test accuracy.

---

## Requirements

- Python 3.x  
- TensorFlow >= 2.x  
- NumPy  
- Matplotlib (optional, for plotting)

---

## Running the Notebooks

Each version is contained in its own `.ipynb` file. For example:

```
cifar_10_classifying_acc41.ipynb  # Version 1
cifar_10_classifying_acc46.ipynb  # Version 2
cifar_10_classifying_acc48.ipynb  # Version 3
cifar_10_classifying_acc74.ipynb  # Version 4
cifar_10_classifying_acc85.ipynb  # Version 5
```

You can open and run each notebook using Jupyter or any compatible environment to train and evaluate the corresponding model.
