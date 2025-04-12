import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load training and testing data
training_df = pd.read_csv('training.csv')
testing_df = pd.read_csv('testing.csv')

# Drop any rows with missing values for simplicity
training_df = training_df.dropna()
testing_df = testing_df.dropna()

# Select top 90 symptoms based on their frequency or relevance
top_symptoms = training_df.drop(columns=['prognosis']).sum().sort_values(ascending=False).index[:90]

# Update training and testing dataframes with only the top symptoms
X_train = training_df[top_symptoms]
y_train = training_df['prognosis']

X_test = testing_df[top_symptoms]
y_test = testing_df['prognosis']

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Calculate accuracy on the test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy:.2f}")

# Function to predict disease based on symptoms
def predict_disease(symptoms):
    # Convert the symptoms into a 90-character string of '0's and '1's based on top_symptoms
    input_string = ''.join('1' if symptom in symptoms else '0' for symptom in top_symptoms)
    
    # Convert the input string to a DataFrame
    input_data = pd.DataFrame([list(map(int, input_string))], columns=top_symptoms)
    
    # Predict the disease
    prediction = model.predict(input_data)
    return prediction[0]

# Function to list symptoms for a disease
def list_symptoms_for_disease(disease_name):
    # Find rows where the disease matches the queried disease name
    disease_rows = training_df[training_df['prognosis'] == disease_name]
    
    # Sum the values of each symptom column to get the count of each symptom associated with the disease
    symptom_counts = disease_rows[top_symptoms].sum()

    # List symptoms where the count is greater than 0 (meaning those symptoms are associated with the disease)
    associated_symptoms = symptom_counts[symptom_counts > 0].index.tolist()

    return associated_symptoms