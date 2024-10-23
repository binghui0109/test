import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from scipy.ndimage import rotate, zoom
from tensorflow.keras import layers
from scipy.ndimage import shift
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns

def rotate_tof_data_2d(data, angle):
    num_samples = data.shape[0]
    rotated_data = np.zeros_like(data)
    
    for i in range(num_samples):
        rotated_frame = rotate(data[i], angle, axes=(0, 1), reshape=False, mode='constant', cval = 0.0)
        rotated_data[i] = np.round(rotated_frame)
    return rotated_data

def translate_tof_data_2d(data, shift_x, shift_y):
    num_samples = data.shape[0]
    translated_data = np.zeros_like(data)
    
    for i in range(num_samples):
        translated_frame = shift(data[i], (shift_y, shift_x, 0), mode='constant', cval = 0.0)
        translated_data[i] = np.round(translated_frame)
    return translated_data

def scale_tof_data(data, scale_factor):
    # Check if the data has the correct shape
    assert data.shape[1:] == (8, 8, 1), "Each sample should be 8x8x3"
    
    # Get the number of samples
    num_samples = data.shape[0]
    
    # Initialize an array to store the scaled data
    scaled_data = np.zeros((num_samples, 8, 8, 1))
    
    for i in range(num_samples):
        # Scale each sample individually
        scaled_sample = zoom(data[i], (scale_factor, scale_factor, 1), order=1)
        scaled_sample = np.round(scaled_sample).astype(np.uint8)
        # Get current dimensions
        current_height, current_width = scaled_sample.shape[:2]
        
        if current_height != 8 or current_width != 8:
            # Calculate padding or cropping
            if current_height < 8:  # Need to pad
                pad_height = 8 - current_height
                pad_top = pad_height // 2
                pad_bottom = pad_height - pad_top
            else:  # Need to crop
                crop_height = current_height - 8
                pad_top = -crop_height // 2
                pad_bottom = -(crop_height - crop_height // 2)
                
            if current_width < 8:  # Need to pad
                pad_width = 8 - current_width
                pad_left = pad_width // 2
                pad_right = pad_width - pad_left
            else:  # Need to crop
                crop_width = current_width - 8
                pad_left = -crop_width // 2
                pad_right = -(crop_width - crop_width // 2)
            
            # Pad or crop as needed
            scaled_sample = np.pad(scaled_sample,
                                  ((max(0, pad_top), max(0, pad_bottom)),
                                   (max(0, pad_left), max(0, pad_right)),
                                   (0, 0)),
                                  mode='edge')
            
            if current_height > 8 or current_width > 8:
                start_h = max(0, -pad_top)
                end_h = start_h + 8
                start_w = max(0, -pad_left)
                end_w = start_w + 8
                scaled_sample = scaled_sample[start_h:end_h, start_w:end_w]
        
        scaled_data[i] = scaled_sample
    
    return scaled_data

def load_and_process_data(directory_path):
    all_data = []
    all_labels = []
    label_encoder = LabelEncoder()

    # Iterate through all CSV files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            class_name = os.path.splitext(filename)[0]  # Get class name from file name

            # Load the space-separated file
            df = pd.read_csv(file_path, header=None, sep='\s+')
            data = df.values
            print(data.shape)
            data = data[:500,:64]
            # data_2 = data[:,128:192]
            # data = np.concatenate((data_1, data_2), axis=1)

            print(f"Shape for {class_name}: {data.shape}")
            # Reshape the data to (num_samples, 32, 8, 8, 1)
            data_1 = data.reshape(-1, 1, 8, 8)
            data = data_1.transpose((0,2,3,1)).astype(np.float64)

            # non_zero_mask = data[:, :, :, 0] != 0
            # data[:, :, :, 0] = np.where(non_zero_mask & (data[:, :, :, 0] > 1800), 0, data[:, :, :, 0])
            # data[:, :, :, 0] = np.where(non_zero_mask & (data[:, :, :, 0] <= 1800), 1, data[:, :, :, 0])
            # data[:, :, :, 1] = np.where(non_zero_mask, data[:, :, :, 1], 0)
            # data[:, :, :, 2] = np.where(non_zero_mask, data[:, :, :, 2], 0)  
            all_data.append(data)
            all_labels.extend([class_name] * data.shape[0])

            print(data[230,:,:,0])
            
            for angle in np.linspace(1, 15, 15):
                rotated_data = rotate_tof_data_2d(data,angle)
                
                # Add random noise to rotated data
                random_array = np.random.random(rotated_data.shape)
                flip_mask = random_array < 0.03                        
                noisy_data  = np.where(flip_mask, 1 - rotated_data, rotated_data)
                all_data.append(noisy_data)
                all_labels.extend([class_name] * noisy_data.shape[0])
                # print(rotated_data[1,:,:,0]) 

            for _ in range(5):
                scale_factor = np.random.uniform(0.8, 1.2)
                scaled_data = scale_tof_data(data, scale_factor)
                # print(scaled_data[230,:,:,0])
                # Randomly decide to flip
                if np.random.random() > 0.5:
                    scaled_data = np.flip(scaled_data, axis = 2)
                if np.random.random() < 0.5:
                    scaled_data = np.flip(scaled_data, axis = 2)   
                all_data.append(scaled_data)
                all_labels.extend([class_name] * scaled_data.shape[0])

            for _ in range(5):
                tx = np.random.uniform(-2, 2)
                ty = np.random.uniform(-2, 2)
                translated_data = translate_tof_data_2d(data, tx, ty)
                all_data.append(translated_data)
                all_labels.extend([class_name] * translated_data.shape[0])
                # print(translated_data[1,:,:,0])

    # Concatenate all data
    data = np.concatenate(all_data, axis=0)
    # Convert string labels to numerical labels
    labels = label_encoder.fit_transform(all_labels)
    data_train, data_test, labels_train, labels_test = train_test_split(data,labels,test_size=0.2,random_state=109)
    # print(data[0,:,:,0])
    # print(data[150,:,:,0])
    # print(data[250,:,:,0])
    # print(data[0,:,:,1])
    # print(data[0,:,:,2])
    # print(data[0,:,:,3])
    print("Final data shape:", data.shape)
    print("Final labels shape:", labels.shape)
    print("Classes:", label_encoder.classes_)
    print("Trainning samples:", data_train.shape)
    print("Testing samples:", data_test.shape)
    return data_train, data_test, labels_train, labels_test, label_encoder
    # return data, labels, label_encoder 

def get_tof_model(data_train, data_test, labels_train, labels_test):
    if len(data_train.shape) != 4 or data_train.shape[1:] != (8, 8, 1):
        raise ValueError(f"Expected data shape (-1, 8, 8, 1), got {data_train.shape}")
    inputs_shape = (8, 8, 1)
    num_classes = len(np.unique(labels_train))
    inputs = tf.keras.Input(shape=inputs_shape)
    labels_train_cat = to_categorical(labels_train, num_classes)
    labels_test_cat = to_categorical(labels_test, num_classes)
    # ---------------------------------------------------------------------------------------
    x = layers.Conv2D(8, (3, 3))(inputs)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="CNN2D_ToF_time_series")
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # Add early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    # Train model
    history = model.fit(
        data_train, labels_train_cat,  # Use categorical labels
        epochs=60,
        validation_split=0.2,
        callbacks=[early_stopping]
    )
    test_loss, test_acc = model.evaluate(data_test, labels_test_cat, verbose=2)  # Use categorical label
    print("Test loss:", test_loss)
    print("Test acc:", test_acc)
    model.summary()
    return model, history

def get_tof_model_multilabel(data_train, data_test, labels_train, labels_test):
    inputs_shape = (8, 8, 3)
    num_classes = len(np.unique(labels_train))
    labels_train = to_categorical(labels_train, num_classes)
    labels_test = to_categorical(labels_test, num_classes)

    inputs = tf.keras.Input(shape=inputs_shape)
    # ---------------------------------------------------------------------------------------
    x = layers.Conv2D(8, (3, 3))(inputs)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="CNN2D_ToF_multilabel")
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(data_train,labels_train, epochs=10)
    test_loss, test_acc = model.evaluate(data_test,labels_test, verbose =2)

    print("Test loss:", test_loss)
    print("Test acc:", test_acc)
    model.summary()
    return model

def plot_confusion(data_test, labels_test, label_encoder):
    class_names = label_encoder.classes_
    labels_pred = model.predict(data_test)
    labels_pred = np.argmax(labels_pred, axis=1)
    confusion_matrix = tf.math.confusion_matrix(labels_test, labels_pred)

    plt.figure()
    sns.heatmap(confusion_matrix,
                annot=True,
                xticklabels=class_names,
                yticklabels=class_names,
                cmap=plt.cm.Blues,
                fmt='d', cbar=False)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()    

    
# Usage
directory_path = 'C:/Users/binghui.lai/ToF_CubeAI/xtacdemo'
# train_directory = 'C:/Users/binghui.lai/ToF_CubeAI/train'
# test_directory = 'C:/Users/binghui.lai/ToF_CubeAI/test'
data_train, data_test, labels_train, labels_test, label_encoder = load_and_process_data(directory_path)
# data_train, labels_train, label_encoder = load_and_process_data(train_directory)
# data_test, labels_test, label_encoder = load_and_process_data(test_directory)
model, history = get_tof_model(data_train, data_test, labels_train, labels_test)
plot_confusion(data_test, labels_test, label_encoder)
model.save('ToF_model_time_series_people_demo.h5')
