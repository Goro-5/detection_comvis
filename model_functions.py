import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

def load_data(directory, img_size=(244, 244)):
    datagen = ImageDataGenerator(rescale=1./255)
    data = []
    labels = []
    class_names = []
    for root, dirs, files in os.walk(directory):
        for name in dirs:
            class_names.append(name)
    class_names.sort()
    class_indices = {name: idx for idx, name in enumerate(class_names)}
    
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        for direction in os.listdir(class_dir):
            direction_dir = os.path.join(class_dir, direction)
            for img_name in os.listdir(direction_dir):
                img_path = os.path.join(direction_dir, img_name)
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size, color_mode='grayscale')
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                data.append(img_array)
                labels.append(class_indices[class_name])
                
    data = np.array(data)
    labels = np.array(labels)
    
    return data, labels, class_names

def create_initial_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def add_noise(data, noise_factor=0.02):
    noise = np.random.normal(loc=0.0, scale=1.0, size=data.shape)
    noisy_data = data + noise_factor * noise
    noisy_data = np.clip(noisy_data, 0.0, 1.0)
    return noisy_data

def train_initial_model(directory, noise_factor=0.02):
    data, labels, class_names = load_data(directory)
    input_shape = data.shape[1:]
    num_classes = len(class_names)
    
    # Add noise to the data
    noisy_data = add_noise(data, noise_factor=noise_factor)
    
    # Split the data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(noisy_data, labels, test_size=0.2, random_state=42)
    
    model = create_initial_model(input_shape, num_classes)
    
    model.fit(train_data, train_labels, epochs=10, validation_data=(val_data, val_labels))
    
    # Evaluate the model
    val_loss, val_acc = model.evaluate(val_data, val_labels)
    print(f"Validation accuracy: {val_acc}")
    
    # Save the model
    model.save('initial_model.h5')
    
    return model

def load_and_predict(model_path, img_path):
    model = load_model(model_path)
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(244, 244), color_mode='grayscale')
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    return predicted_class

def update_model(model_path, directory, noise_factor=0.02):
    model = load_model(model_path)
    data, labels, _ = load_data(directory)
    
    # Add noise to the data
    noisy_data = add_noise(data, noise_factor=noise_factor)
    
    # Split the data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(noisy_data, labels, test_size=0.2, random_state=42)
    
    model.fit(train_data, train_labels, epochs=10, validation_data=(val_data, val_labels))
    
    # Evaluate the model
    val_loss, val_acc = model.evaluate(val_data, val_labels)
    print(f"Validation accuracy after update: {val_acc}")
    
    # Save the updated model
    model.save('updated_model.h5')
    
    return model

def expand_model_for_new_classes(model_path, new_classes_directory, noise_factor=0.02):
    model = load_model(model_path)
    old_data, old_labels, old_class_names = load_data(os.path.dirname(model_path))
    new_data, new_labels, new_class_names = load_data(new_classes_directory)
    
    # Combine old and new data
    combined_data = np.concatenate((old_data, new_data))
    combined_labels = np.concatenate((old_labels, new_labels))
    combined_class_names = sorted(list(set(old_class_names + new_class_names)))
    
    # Add noise to the data
    noisy_data = add_noise(combined_data, noise_factor=noise_factor)
    
    # Split the data into training and validation sets
    train_data, val_data, train_labels, val_labels = train_test_split(noisy_data, combined_labels, test_size=0.2, random_state=42)
    
    # Adjust the model for new classes
    num_classes = len(combined_class_names)
    model.layers[-1] = tf.keras.layers.Dense(num_classes, activation='softmax')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(train_data, train_labels, epochs=10, validation_data=(val_data, val_labels))
    
    # Evaluate the model
    val_loss, val_acc = model.evaluate(val_data, val_labels)
    print(f"Validation accuracy after expanding classes: {val_acc}")
    
    # Save the expanded model
    model.save('expanded_model.h5')
    
    return model
