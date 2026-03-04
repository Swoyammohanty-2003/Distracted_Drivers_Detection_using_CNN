# Setup environment
!pip install kaggle tensorflow opencv-python-headless numpy visualkeras keras-tuner

# Enhanced Distracted Driver Detection
# Adding Hyperparameter Tuning, Transfer Learning, and K-Fold Validation

# Import necessary libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import random
from PIL import Image
import keras_tuner as kt
import visualkeras
import shutil

# Setup Kaggle API
from google.colab import files
files.upload()  # Upload kaggle.json

! mkdir -p ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json

# Download the dataset
! kaggle competitions download -c state-farm-distracted-driver-detection --force

# Extract the dataset
import zipfile
with zipfile.ZipFile('state-farm-distracted-driver-detection.zip', 'r') as zip_ref:
    zip_ref.extractall('state_farm_dataset')

# Load driver information
df_driver = pd.read_csv('/content/state_farm_dataset/driver_imgs_list.csv')

# Create a reduced dataset for faster experimentation
reduced_dataset_path = "/content/state_farm_dataset/reduced_train"
os.makedirs(reduced_dataset_path, exist_ok=True)

# Select a subset of images for each class
selected_images = df_driver.groupby("classname").apply(lambda x: x.sample(n=200, random_state=42)).reset_index(drop=True)

# Copy images to reduced dataset
original_train_path = "/content/state_farm_dataset/imgs/train"

for index, row in selected_images.iterrows():
    class_folder = row["classname"]
    img_name = row["img"]

    src = os.path.join(original_train_path, class_folder, img_name)
    dst_dir = os.path.join(reduced_dataset_path, class_folder)
    dst = os.path.join(dst_dir, img_name)

    os.makedirs(dst_dir, exist_ok=True)
    if os.path.exists(src):
        shutil.copy(src, dst)

print("✅ Dataset reduced successfully!")

# Define class names for reference
class_names = [
    "Safe Driving",
    "Texting Right",
    "Talking on Phone - Right",
    "Texting Left",
    "Talking on Phone - Left",
    "Operating Radio",
    "Drinking",
    "Reaching Behind",
    "Hair & Makeup",
    "Talking to Passenger"
]

# Classes dictionary
classes_dict = {
    0: "Safe Driving",
    1: "Texting Right",
    2: "Talking on Phone - Right",
    3: "Texting Left",
    4: "Talking on Phone - Left",
    5: "Operating Radio",
    6: "Drinking",
    7: "Reaching Behind",
    8: "Hair & Makeup",
    9: "Talking to Passenger"
}

# Define image dimensions
IMG_SIZE = 224  # Increased for MobileNetV2
BATCH_SIZE = 32

# Part 1: Visualize Dataset Distribution
def visualize_dataset_distribution(df, train_dir):
    class_counts = df['classname'].value_counts()
    plt.figure(figsize=(10, 5))
    plt.bar(class_counts.index, class_counts.values)
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.title("Number of Images per Class")
    plt.xticks(rotation=45)
    plt.show()

    if 'subject' in df.columns:
        driver_counts = df['subject'].value_counts()
        plt.figure(figsize=(10, 5))
        plt.bar(driver_counts.index.astype(str), driver_counts.values)
        plt.xlabel("Driver")
        plt.ylabel("Number of Images")
        plt.title("Number of Images per Driver")
        plt.show()
    else:
        print("Column 'subject' not found. Skipping driver distribution plot.")

# Part 2: Preview Sample Images
def preview_training_images(data_dir, num_rows=3, num_cols=5):
    all_images = []
    all_labels = []
    classes = os.listdir(data_dir)
    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        if os.path.isdir(cls_dir):
            imgs = os.listdir(cls_dir)
            for img in imgs:
                all_images.append(os.path.join(cls_dir, img))
                all_labels.append(cls)

    num_images = num_rows * num_cols
    indices = random.sample(range(len(all_images)), num_images)
    selected_images = [all_images[i] for i in indices]
    selected_labels = [all_labels[i] for i in indices]

    plt.figure(figsize=(15, 9))
    for i, (img_path, label) in enumerate(zip(selected_images, selected_labels)):
        img = Image.open(img_path)
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(img)
        plt.title(label)
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Part 3: Create Base CNN Model
def create_base_model(img_rows, img_cols, color_type):
    model = Sequential()

    # CNN Layer 1
    model.add(Conv2D(16, (3,3), activation='relu', input_shape=(img_rows, img_cols, color_type)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    # CNN Layer 2
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    # CNN Layer 3
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.3))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model

# Part 4: Create Transfer Learning Model with MobileNetV2
def create_transfer_learning_model():
    # Load MobileNetV2 with pre-trained weights
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze base model layers
    #base_model.trainable = False
    base_model.trainable = True
    for layer in base_model.layers[:80]:  # Freeze fewer layers
        layer.trainable = False

    # Create the model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])

    return model

# Part 5: Hyperparameter Tuning Model Builder
def build_tunable_model(hp):
    # Load MobileNetV2 with pre-trained weights
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze base model layers
    base_model.trainable = False

    # Create the model
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())

    # Tune number of dense layers and units
    for i in range(hp.Int('num_dense_layers', 1, 3)):
        model.add(Dense(
            units=hp.Int(f'dense_{i}_units', min_value=128, max_value=512, step=64),
            activation='relu'
        ))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float(f'dropout_{i}', min_value=0.2, max_value=0.5, step=0.1)))

    # Output layer
    model.add(Dense(10, activation='softmax'))

    # Tune learning rate
    learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3, 5e-3])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Part 6: Plot Training History
def plot_training_history(history):
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(12, 5))
    ax[0].plot(history.history['loss'], label='Training')
    ax[0].plot(history.history['val_loss'], label='Validation')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].set_title('Training & Validation Loss')

    ax[1].plot(history.history['accuracy'], label='Training')
    ax[1].plot(history.history['val_accuracy'], label='Validation')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    ax[1].set_title('Training & Validation Accuracy')

    plt.tight_layout()
    plt.show()

# Part 7: Print Evaluation Metrics
def print_evaluation_metrics(y_true, y_pred):
    # Calculate and display accuracy
    acc = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {acc*100:.2f}%")

    # Generate detailed classification report
    print("\nDetailed Classification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Convert decimal values to percentages
    percent_df = report_df.copy()
    for col in ['precision', 'recall', 'f1-score']:
        percent_df[col] = percent_df[col] * 100

    print(percent_df.round(2).to_string())

    # Create and plot confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Calculate normalized confusion matrix
    print("\nNormalized Confusion Matrix (percentages):")
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.1f', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix (%)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Calculate per-class accuracy
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {per_class_acc[i]:.2f}%")

    # Print summary
    print("\nSummary:")
    print(f"Evaluated Accuracy: {acc*100:.2f}%")
    print(f"F1-Score (Weighted Average): {report['weighted avg']['f1-score']*100:.2f}%")
    print(f"Precision (Weighted Average): {report['weighted avg']['precision']*100:.2f}%")
    print(f"Recall (Weighted Average): {report['weighted avg']['recall']*100:.2f}%")

# Part 8: K-Fold Cross-Validation
def perform_kfold_validation(model_type='transfer', n_splits=5):
    # Create a directory to store images for K-Fold validation
    kfold_data_dir = '/content/state_farm_dataset/kfold_data'
    os.makedirs(kfold_data_dir, exist_ok=True)

    # Copy images to K-Fold directory
    for class_name in os.listdir(reduced_dataset_path):
        class_dir = os.path.join(reduced_dataset_path, class_name)
        if os.path.isdir(class_dir):
            kfold_class_dir = os.path.join(kfold_data_dir, class_name)
            os.makedirs(kfold_class_dir, exist_ok=True)
            for img_name in os.listdir(class_dir):
                src = os.path.join(class_dir, img_name)
                dst = os.path.join(kfold_class_dir, img_name)
                if not os.path.exists(dst):
                    shutil.copy(src, dst)

    # Get all image paths and labels
    all_images = []
    all_labels = []
    for class_idx, class_name in enumerate(sorted(os.listdir(kfold_data_dir))):
        class_dir = os.path.join(kfold_data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                all_images.append(os.path.join(class_dir, img_name))
                all_labels.append(class_idx)

    # Convert to numpy arrays
    all_images = np.array(all_images)
    all_labels = np.array(all_labels)

    # Initialize KFold
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # For storing metrics
    fold_accuracies = []
    fold_val_accuracies = []

    # Initialize figure for plotting loss and accuracy
    fig, axes = plt.subplots(n_splits, 2, figsize=(15, n_splits*5))

    # Initialize callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )

    # Loop through each fold
    for fold, (train_indices, val_indices) in enumerate(kfold.split(all_images)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")

        # Get training and validation data for this fold
        X_train, X_val = all_images[train_indices], all_images[val_indices]
        y_train, y_val = all_labels[train_indices], all_labels[val_indices]

        # Create directories for this fold
        fold_train_dir = f'/content/state_farm_dataset/fold_{fold+1}_train'
        fold_val_dir = f'/content/state_farm_dataset/fold_{fold+1}_val'

        os.makedirs(fold_train_dir, exist_ok=True)
        os.makedirs(fold_val_dir, exist_ok=True)

        # Clear previous fold directories
        for d in [fold_train_dir, fold_val_dir]:
            for class_name in os.listdir(kfold_data_dir):
                os.makedirs(os.path.join(d, class_name), exist_ok=True)

        # Copy training images to fold directory
        for img_path, label in zip(X_train, y_train):
            class_name = f'c{label}'
            dst_dir = os.path.join(fold_train_dir, class_name)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, os.path.basename(img_path))
            shutil.copy(img_path, dst)

        # Copy validation images to fold directory
        for img_path, label in zip(X_val, y_val):
            class_name = f'c{label}'
            dst_dir = os.path.join(fold_val_dir, class_name)
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, os.path.basename(img_path))
            shutil.copy(img_path, dst)

        # Create data generators
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        val_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            fold_train_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        val_generator = val_datagen.flow_from_directory(
            fold_val_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical'
        )

        # Create model based on model_type
        if model_type == 'transfer':
            model = create_transfer_learning_model()
        else:
            model = create_base_model(IMG_SIZE, IMG_SIZE, 3)

        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train model
        history = model.fit(
            train_generator,
            epochs=30,
            validation_data=val_generator,
            callbacks=[early_stopping, reduce_lr]
        )

        # Store metrics
        fold_accuracies.append(max(history.history['accuracy']))
        fold_val_accuracies.append(max(history.history['val_accuracy']))

        # Plot training history for this fold
        axes[fold, 0].plot(history.history['loss'], label='Training')
        axes[fold, 0].plot(history.history['val_loss'], label='Validation')
        axes[fold, 0].set_title(f'Fold {fold+1} Loss')
        axes[fold, 0].set_xlabel('Epochs')
        axes[fold, 0].set_ylabel('Loss')
        axes[fold, 0].legend()

        axes[fold, 1].plot(history.history['accuracy'], label='Training')
        axes[fold, 1].plot(history.history['val_accuracy'], label='Validation')
        axes[fold, 1].set_title(f'Fold {fold+1} Accuracy')
        axes[fold, 1].set_xlabel('Epochs')
        axes[fold, 1].set_ylabel('Accuracy')
        axes[fold, 1].legend()

        # Evaluate model on validation set
        print(f"\nEvaluating fold {fold+1}...")

        val_generator.reset()
        y_true = val_generator.classes
        steps = len(val_generator)

        y_pred_probs = model.predict(val_generator, steps=steps)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Print classification report for this fold
        print(f"Fold {fold+1} Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))

        # Clean up fold directories to save space
        shutil.rmtree(fold_train_dir)
        shutil.rmtree(fold_val_dir)

    # Adjust layout and show plots
    plt.tight_layout()
    plt.savefig('kfold_history.png')
    plt.show()

    # Print summary of all folds
    print("\n--- K-Fold Cross-Validation Summary ---")
    for fold in range(n_splits):
        print(f"Fold {fold+1}: Training Accuracy = {fold_accuracies[fold]*100:.2f}%, Validation Accuracy = {fold_val_accuracies[fold]*100:.2f}%")

    print(f"\nAverage Training Accuracy: {np.mean(fold_accuracies)*100:.2f}%")
    print(f"Average Validation Accuracy: {np.mean(fold_val_accuracies)*100:.2f}%")
    print(f"Standard Deviation of Validation Accuracy: {np.std(fold_val_accuracies)*100:.2f}%")

    return np.mean(fold_val_accuracies)

# Part 9: Hyperparameter Tuning with Keras Tuner
def perform_hyperparameter_tuning():
    print("\n--- Starting Hyperparameter Tuning ---")

    # Create data generators
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        reduced_dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        reduced_dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Create the tuner
    tuner = kt.RandomSearch(
        build_tunable_model,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='hyperparameter_tuning',
        project_name='distracted_driver'
    )

    # Create callbacks for early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # Search for best hyperparameters
    tuner.search(
        train_gen,
        epochs=15,
        validation_data=val_gen,
        callbacks=[early_stopping]
    )

    # Get best hyperparameters
    best_hps = tuner.get_best_hyperparameters(1)[0]
    print("\nBest Hyperparameters:")
    print(f"Learning Rate: {best_hps.get('learning_rate')}")
    print(f"Number of Dense Layers: {best_hps.get('num_dense_layers')}")

    for i in range(best_hps.get('num_dense_layers')):
        print(f"Dense Layer {i+1} Units: {best_hps.get(f'dense_{i}_units')}")
        print(f"Dropout {i+1} Rate: {best_hps.get(f'dropout_{i}')}")

    # Build best model
    best_model = tuner.hypermodel.build(best_hps)

    # Train best model
    history = best_model.fit(
        train_gen,
        epochs=30,
        validation_data=val_gen,
        callbacks=[early_stopping]
    )

    # Plot training history
    plot_training_history(history)

    return best_model, history

# Part 10: Fine-tune Transfer Learning Model
def fine_tune_transfer_model(model):
    print("\n--- Fine-tuning Transfer Learning Model ---")

    # Create data generators with augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        reduced_dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        reduced_dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Unfreeze some layers of the base model
    base_model = model.layers[0]
    base_model.trainable = True

    # Freeze first 100 layers, unfreeze the rest
    for layer in base_model.layers[:100]:
        layer.trainable = False

    # Recompile model with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Create callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )

    # Fine-tune model
    history = model.fit(
        train_gen,
        epochs=30,
        validation_data=val_gen,
        callbacks=[early_stopping, reduce_lr]
    )

    # Plot training history
    plot_training_history(history)

    return model, history

# Part 11: Evaluate Model
def evaluate_model(model, display_results=True):
    # Create data generator
    datagen = ImageDataGenerator(rescale=1./255)

    # Generate validation data
    val_gen = datagen.flow_from_directory(
        reduced_dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # Evaluate model on validation set
    val_gen.reset()
    y_true = val_gen.classes
    steps = len(val_gen)

    y_pred_probs = model.predict(val_gen, steps=steps)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Print evaluation metrics
    if display_results:
        print_evaluation_metrics(y_true, y_pred)

    # Calculate accuracy
    acc = accuracy_score(y_true, y_pred)

    return acc

# Part 12: Visualize Model Architecture
def visualize_model_architecture(model):
    try:
        visualkeras.layered_view(model, legend=True).show()
        tf.keras.utils.plot_model(
            model,
            to_file='model_architecture.png',
            show_shapes=True,
            show_layer_names=True,
            rankdir='TB',
            expand_nested=True,
            dpi=96
        )
        print("Model architecture visualized and saved as 'model_architecture.png'")
    except Exception as e:
        print(f"Error visualizing model architecture: {e}")



# Main execution flow
def main():
    # ✅ Save and download untrained model for UI testing
    #print("\n--- Saving Untrained Model for UI Testing ---")
    #model = create_transfer_learning_model()
    #model.save("best_driver_detection_model.keras")

    #from google.colab import files
    #files.download("best_driver_detection_model.keras")
    #return  # Skip rest of main for this test
    #print("Starting Enhanced Distracted Driver Detection...")

    # 1. Visualize dataset distribution
    visualize_dataset_distribution(df_driver, reduced_dataset_path)

    # 2. Preview sample images
    preview_training_images(reduced_dataset_path)

    # 3. Create a transfer learning model with MobileNetV2
    print("\n--- Creating Transfer Learning Model with MobileNetV2 ---")
    transfer_model = create_transfer_learning_model()
    transfer_model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Create data generators with augmentation for transfer learning
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        brightness_range=[0.7, 1.3],
        horizontal_flip=True,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        reduced_dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        reduced_dataset_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # Train the transfer learning model
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6
    )

    # Train transfer learning model
    class_weight = {i: 1.0 for i in range(10)}
    class_weight[8] = 2.5
    transfer_history = transfer_model.fit(
        train_gen,
        epochs=30,
        validation_data=val_gen,
        callbacks=[early_stopping, reduce_lr]
    )

    # Plot training history
    plot_training_history(transfer_history)

    # Evaluate transfer learning model
    print("\n--- Evaluating Transfer Learning Model ---")
    transfer_accuracy = evaluate_model(transfer_model)

    # 4. Perform K-Fold Cross-Validation with transfer learning
    print("\n--- Performing K-Fold Cross-Validation with Transfer Learning ---")
    kfold_accuracy = perform_kfold_validation(model_type='transfer', n_splits=5)

    # 5. Hyperparameter Tuning
    best_model, tuning_history = perform_hyperparameter_tuning()

    # Evaluate best model from hyperparameter tuning
    print("\n--- Evaluating Best Model from Hyperparameter Tuning ---")
    best_model_accuracy = evaluate_model(best_model)

    # 6. Fine-tune transfer learning model
    fine_tuned_model, fine_tuning_history = fine_tune_transfer_model(transfer_model)

    # Evaluate fine-tuned model
    print("\n--- Evaluating Fine-Tuned Model ---")
    fine_tuned_accuracy = evaluate_model(fine_tuned_model)

    # 7. Visualize model architecture
    print("\n--- Visualizing Model Architecture ---")
    visualize_model_architecture(fine_tuned_model)

    # 8. Compare all models
    print("\n--- Model Comparison ---")
    print(f"Transfer Learning Model Accuracy: {transfer_accuracy*100:.2f}%")
    print(f"K-Fold Cross-Validation Average Accuracy: {kfold_accuracy*100:.2f}%")
    print(f"Best Model from Hyperparameter Tuning Accuracy: {best_model_accuracy*100:.2f}%")
    print(f"Fine-Tuned Model Accuracy: {fine_tuned_accuracy*100:.2f}%")

    # 9. Save the best model
    print("\n--- Saving Best Model ---")
    best_overall_model = fine_tuned_model  # Assuming fine-tuned model is the best
    best_overall_model.save("best_driver_detection_model.keras")

    # Trigger model download in Colab
    from google.colab import files
    files.download("best_driver_detection_model.keras")

    # Allow prediction on uploaded images
    # print("\n--- Test the model on your own images ---")
    # print("Upload an image to predict the driver's state:")
    # predict_single_image(best_overall_model)

    print("\nEnhanced Distracted Driver Detection completed successfully!")

# Run the main function
if __name__ == "__main__":
    main()