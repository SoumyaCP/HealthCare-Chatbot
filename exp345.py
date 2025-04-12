import pandas as pd
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score

# Load training data (assuming 'training.csv' is your dataset filename)
training_df = pd.read_csv('training.csv')

# Drop any rows with missing values for simplicity
training_df = training_df.dropna()

# Split training data into features (symptoms) and target (prognosis labels)
X_train = training_df.drop(columns=['prognosis'])  # Features are all columns except 'prognosis'
y_train = training_df['prognosis']  # Target variable is 'prognosis'

# Train a Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Load the CSV file (assuming 'testing.csv' is your dataset filename)
file_path = 'testing.csv'
df = pd.read_csv(file_path)

# Create a column to count the number of symptoms with a value of 1 for each disease
df['symptom_count'] = df.drop(columns=['prognosis']).sum(axis=1)

# Filter the dataframe for diseases with more than 5 symptoms
df_filtered = df[df['symptom_count'] > 5]

# Check if there are at least 30 diseases in df_filtered
if len(df_filtered) < 30:
    print(f"Warning: There are only {len(df_filtered)} diseases with more than 5 symptoms.")

# Select random diseases from the filtered dataframe
if len(df_filtered) >= 30:
    df_random_30 = df_filtered.sample(n=30, random_state=1)  # random_state for reproducibility
else:
    df_random_30 = df_filtered  # Sample all available diseases if less than 30

# Function to predict diseases based on a specified number of symptoms
def predict_diseases(symptoms_count, df_random_30):
    # Create a dictionary to store diseases and their corresponding symptoms
    disease_symptoms_all = {}
    symptom_columns_all = df.columns.drop(['prognosis', 'symptom_count'])

    for _, row in df_random_30.iterrows():
        disease = row['prognosis']
        symptoms = [symptom for symptom in symptom_columns_all if row[symptom] == 1]
        # Select specified number of random symptoms
        random.shuffle(symptoms)
        symptoms = symptoms[:symptoms_count]  # Take the first `symptoms_count` symptoms (randomly shuffled)
        disease_symptoms_all[disease] = symptoms

    # Initialize counters for correct and incorrect predictions
    correct_predictions = 0
    incorrect_predictions = 0

    # Loop through each disease and predict based on specified number of symptoms
    for disease, symptoms in disease_symptoms_all.items():
        # Convert the symptoms into a 132-character string of '0's and '1's
        all_symptoms = X_train.columns
        input_string = ''.join('1' if symptom in symptoms else '0' for symptom in all_symptoms)
        
        # Convert the input string to a DataFrame
        input_data = pd.DataFrame([list(map(int, input_string))], columns=all_symptoms)
        
        # Predict the disease
        predicted_disease = model.predict(input_data)[0]
        
        # Check if the predicted disease matches the disease in the list
        if predicted_disease == disease:
            correct_predictions += 1
        else:
            incorrect_predictions += 1

    # Calculate accuracy and number of correct predictions
    accuracy = correct_predictions / (correct_predictions + incorrect_predictions)
    return accuracy, correct_predictions

# Perform three rounds of experiments with 3, 4, and 5 symptoms
experiment_results = {}

for round_num in range(1, 4):
    round_results = {}
    for symptoms_count in [3, 4, 5]:
        accuracy, correct_count = predict_diseases(symptoms_count, df_random_30)
        round_results[f"{symptoms_count} Symptoms"] = {'Accuracy': accuracy, 'Correct Predictions': correct_count}
    
    experiment_results[f"Round {round_num}"] = round_results

# Print results in a table format for each round
for round_name, round_result in experiment_results.items():
    print(f"\nExperiment Results - {round_name}:")
    results_data = []
    for symptoms_count, metrics in round_result.items():
        accuracy = metrics['Accuracy']
        correct_count = metrics['Correct Predictions']
        results_data.append([symptoms_count, accuracy, correct_count])
    
    results_df = pd.DataFrame(results_data, columns=['Symptoms Count', 'Accuracy', 'Correct Predictions'])
    print(results_df)

# Plotting the results grouped by rounds
plt.figure(figsize=(10, 6))
bar_width = 0.25
index = range(len([3, 4, 5]))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # blue, orange, green
legend_labels = []

for i, symptoms_count in enumerate([3, 4, 5]):
    round_accuracies = [round_result[f"{symptoms_count} Symptoms"]['Accuracy'] for round_result in experiment_results.values()]
    plt.bar([idx + i * bar_width for idx in index], round_accuracies, bar_width, color=colors[i])
    legend_labels.append(f"{symptoms_count} Symptoms")

plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.title('Prediction Accuracy by Round')
plt.xticks([idx + bar_width for idx in index], [f"Round {round_num}" for round_num in range(1, 4)])
plt.ylim(0, 1)  # Set y-axis limit to 0-100% for accuracy
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(legend_labels, loc='upper right')
plt.show()
