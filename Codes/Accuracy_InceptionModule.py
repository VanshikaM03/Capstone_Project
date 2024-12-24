
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

# Load the fused cleaned file
fused_file_path = r"C:\Users\selin\Desktop\Correct_Data\Fused_Cleaned.csv"  # Update the file path
data = pd.read_csv(fused_file_path)

# Separate features and labels
X = data.iloc[:, :-1].to_numpy()  # All columns except the last one
y = data.iloc[:, -1].to_numpy()  # Last column (labels)

# One-hot encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_one_hot = to_categorical(y_encoded)

# Reshape the features to include a channel dimension (needed for Conv1D)
X = X.reshape(X.shape[0], X.shape[1], 1)


# Define the Inception module
def inception_module(input_layer, filters):
    # 1x1 Convolution
    conv1 = layers.Conv1D(filters, 1, padding='same', activation='relu')(input_layer)

    # 3x3 Convolution
    conv3 = layers.Conv1D(filters, 3, padding='same', activation='relu')(input_layer)

    # 5x5 Convolution
    conv5 = layers.Conv1D(filters, 5, padding='same', activation='relu')(input_layer)

    # MaxPooling followed by 1x1 Convolution
    maxpool = layers.MaxPooling1D(pool_size=3, strides=1, padding='same')(input_layer)
    maxpool_conv = layers.Conv1D(filters, 1, padding='same', activation='relu')(maxpool)

    # Concatenate all the paths
    output = layers.concatenate([conv1, conv3, conv5, maxpool_conv], axis=-1)
    return output


# Build the Inception-based model
def build_inception_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Inception module stack
    x = inception_module(inputs, 32)
    x = inception_module(x, 64)

    # Global Average Pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Fully connected layers
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)  # Dropout for regularization
    outputs = layers.Dense(num_classes, activation='softmax')(x)  # Softmax layer for multiclass classification

    # Define the model
    model = Model(inputs, outputs)
    return model


# Perform 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
accuracy_per_fold = []
loss_per_fold = []
training_loss_per_fold = []
validation_loss_per_fold = []
training_accuracy_per_fold = []
validation_accuracy_per_fold = []

input_shape = (X.shape[1], 1)
num_classes = y_one_hot.shape[1]

for train_index, test_index in kf.split(X):
    print(f"Training on Fold {fold}...")

    # Split data for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_one_hot[train_index], y_one_hot[test_index]

    # Build and compile the model
    model = build_inception_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Fold {fold} - Loss: {test_loss}, Accuracy: {test_accuracy}")

    # Store results
    accuracy_per_fold.append(test_accuracy)
    loss_per_fold.append(test_loss)
    training_loss_per_fold.append(history.history['loss'])
    validation_loss_per_fold.append(history.history['val_loss'])
    training_accuracy_per_fold.append(history.history['accuracy'])
    validation_accuracy_per_fold.append(history.history['val_accuracy'])

    fold += 1

# Calculate average accuracy and loss across all folds
average_accuracy = np.mean(accuracy_per_fold)
average_loss = np.mean(loss_per_fold)

print(f"\nAverage Test Accuracy: {average_accuracy:.2f}")
print(f"Average Test Loss: {average_loss:.2f}")

# Plot the loss curve
plt.figure(figsize=(12, 6))
for i in range(len(training_loss_per_fold)):
    plt.plot(training_loss_per_fold[i], label=f'Fold {i + 1} - Training Loss')
    plt.plot(validation_loss_per_fold[i], linestyle='--', label=f'Fold {i + 1} - Validation Loss')

plt.title("Training and Validation Loss Across Folds")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()

# Plot the accuracy curve
plt.figure(figsize=(12, 6))
for i in range(len(training_accuracy_per_fold)):
    plt.plot(training_accuracy_per_fold[i], label=f'Fold {i + 1} - Training Accuracy')
    plt.plot(validation_accuracy_per_fold[i], linestyle='--', label=f'Fold {i + 1} - Validation Accuracy')

plt.title("Training and Validation Accuracy Across Folds")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()




