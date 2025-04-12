import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import numpy as np

# Load training and testing data
training_df = pd.read_csv('training.csv')
testing_df = pd.read_csv('testing.csv')

# Drop any rows with missing values for simplicity
training_df = training_df.dropna()
testing_df = testing_df.dropna()

# Split training data into features (symptoms) and target (prognosis labels)
X_train = training_df.drop(columns=['prognosis'])  # Features are all columns except 'prognosis'
y_train = training_df['prognosis']  # Target variable is 'prognosis'

# Split testing data into features (symptoms) and target (prognosis labels)
X_test = testing_df.drop(columns=['prognosis'])  # Features are all columns except 'prognosis'
y_test = testing_df['prognosis']  # Target variable is 'prognosis'

# Initialize the Naive Bayes model
model = MultinomialNB()

# Number of epochs
epochs = 50

# Lists to store accuracy values for each epoch
train_accuracies = []
test_accuracies = []

for epoch in range(epochs):
    # Train the model incrementally
    model.partial_fit(X_train, y_train, classes=np.unique(y_train))
    
    # Calculate training accuracy
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_accuracies.append(train_accuracy)
    
    # Calculate testing accuracy
    y_test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_accuracies.append(test_accuracy)

# Plot the accuracies over epochs
plt.figure(figsize=(14, 6))

# Plot 1: Training and Testing Accuracy over Epochs
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_accuracies, label='Training accuracy', color='blue')
plt.plot(range(1, epochs + 1), test_accuracies, label='Testing accuracy', color='red')
plt.title('Training and Testing Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot 2: Training and Testing Accuracy over Epochs (duplicate for better comparison)
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_accuracies, label='Training accuracy', color='blue')
plt.plot(range(1, epochs + 1), test_accuracies, label='Testing accuracy', color='red')
plt.title('Training and Testing Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Display final accuracies
print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}")
print(f"Final Testing Accuracy: {test_accuracies[-1]:.2f}")
