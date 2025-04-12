import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from joblib import dump

# Load training dataset from CSV file
training_df = pd.read_csv('Training.csv')

# Drop any rows with missing values for simplicity
training_df = training_df.dropna()

# Split training data into features (symptoms) and target (prognosis labels)
X_train = training_df.drop(columns=['prognosis'])  # Features are all columns except 'prognosis'
y_train = training_df['prognosis']  # Target variable is 'prognosis'

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the trained model to a file
dump(model, 'trained_model.joblib')

print("Model training complete and model saved to 'trained_model.joblib'")