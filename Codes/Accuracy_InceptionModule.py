import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

# Load the fused cleaned file
fused_file_path = r"C:\Users\selin\Desktop\Cleaned\Fused_Cleaned5.csv"
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
conf_matrix_total = np.zeros((y_one_hot.shape[1], y_one_hot.shape[1]))  # Initialize confusion matrix for all folds
classification_reports = []
kappa_scores = []

input_shape = (X.shape[1], 1)
num_classes = y_one_hot.shape[1]

for train_index, test_index in kf.split(X):
    print(f"Training on Fold {fold}...")

    # Split data for this fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y_one_hot[train_index], y_one_hot[test_index]
    y_test_class = y_encoded[test_index]  # for classification report

    # Build and compile the model
    model = build_inception_model(input_shape, num_classes)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=150,
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

    # Predict on the test set
    y_pred = model.predict(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)  # Get the class with the highest probability

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test_class, y_pred_class)
    conf_matrix_total += conf_matrix

    # Classification Report with labels
    report = classification_report(y_test_class, y_pred_class, target_names=label_encoder.classes_, output_dict=True)
    classification_reports.append(report)

    # Calculate Cohen's Kappa
    kappa = cohen_kappa_score(y_test_class, y_pred_class)
    kappa_scores.append(kappa)

    fold += 1

# Calculate average accuracy and loss across all folds
average_accuracy = np.mean(accuracy_per_fold)
average_loss = np.mean(loss_per_fold)

print(f"\nAverage Test Accuracy: {average_accuracy:.4f}")
print(f"Average Test Loss: {average_loss:.4f}")

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

# Aggregate classification report and print
print("\nAverage Classification Report for All Folds:")

# Aggregate results per metric (precision, recall, f1-score)
labels = label_encoder.classes_
aggregated_report = {label: {'precision': [], 'recall': [], 'f1-score': []} for label in labels}

for report in classification_reports:
    for label in labels:
        aggregated_report[label]['precision'].append(report[label]['precision'])
        aggregated_report[label]['recall'].append(report[label]['recall'])
        aggregated_report[label]['f1-score'].append(report[label]['f1-score'])

# Calculate mean and display it
avg_classification_report = {
    label: {
        'precision': np.mean(aggregated_report[label]['precision']),
        'recall': np.mean(aggregated_report[label]['recall']),
        'f1-score': np.mean(aggregated_report[label]['f1-score']),
    }
    for label in labels
}

# Display as a DataFrame for better readability
print("\nAggregated Classification Report (Mean across folds):")
print(pd.DataFrame(avg_classification_report).T)

# Display confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix_total, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix Across All Folds')
plt.colorbar()
tick_marks = np.arange(len(labels))
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)

# Plotting text on the confusion matrix
thresh = conf_matrix_total.max() / 2.
for i, j in np.ndindex(conf_matrix_total.shape):
    plt.text(j, i, format(conf_matrix_total[i, j], '0.2f'),  # Use '0.2f' for float formatting
             horizontalalignment="center",
             color="white" if conf_matrix_total[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()

# Cohen's Kappa scores
print(f"Cohen's Kappa Scores for All Folds: {kappa_scores}")





