import spacy
import pandas as pd
import re
import PyPDF2
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from flask_mysqldb import MySQL
import MySQLdb.cursors
from symptom_extractor1 import extract_symptoms
from disease_predictor1 import predict_disease
import MySQLdb
import wikipediaapi
import requests
from bs4 import BeautifulSoup

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Load the dataset
df = pd.read_csv('medicines.csv')

# Initialize Wikipedia API with a user agent
wiki_wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='YourAppName/1.0 (contact@example.com)'
)

# Function to fetch summary from Wikipedia
def fetch_wikipedia_summary(disease_name):
    page = wiki_wiki.page(disease_name)
    print(f"Fetching Wikipedia page for: {disease_name}")
    if page.exists():
        summary = page.summary
        print(f"Page found. Summary length: {len(summary)}")
        sentences = summary.split('. ')
        if len(sentences) > 3:
            return '. '.join(sentences[:3]) + '.'
        else:
            return '. '.join(sentences) + '.'
    else:
        return 0


# Function to fetch disease summary from multiple sources
def fetch_disease_summary(disease_name):
    simplified_name = re.sub(r'\(.*?\)', '', disease_name).strip()
    print(f"Searching for: {simplified_name}")

    summary = fetch_wikipedia_summary(simplified_name)
    if summary:
        return f"Information from Wikipedia:\n{summary}"



    return 0

# Function to suggest drug for a given disease (matching as a substring)
def suggest_drug_for_disease(df, disease_name):
    disease_name_lower = disease_name.lower().strip()
    drugs = df[df['disease'].str.lower().str.contains(disease_name_lower)]['drug'].tolist()
    if drugs:
        return list(set(drugs))  # Return unique drugs
    else:
        return ["No drug found for the given disease"]

# Function to find disease associated with a given drug
def find_disease_for_drug(df, drug_name):
    drug_name_lower = drug_name.lower().strip()
    diseases = df[df['drug'].str.lower().apply(lambda drugs: any(drug_name_lower in drug for drug in drugs.split(' / ')))]['disease'].tolist()
    if diseases:
        return list(set(diseases))  # Return unique diseases
    else:
        return ["No disease found for the given drug"]

def extract_hospitals_by_pincode(pdf_path, target_pincode):
    hospitals = []

    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text = page.extract_text()
            lines = text.split('\n')
            
            for line in lines:
                if re.search(fr'\b{target_pincode}\b', line):
                    # Capture the full hospital information including the pincode
                    match = re.search(fr'(.+?)\s*\b{target_pincode}\b', line)
                    if match:
                        hospital_info = match.group(1).strip()+"<br>"
                        hospitals.append(hospital_info)
    
    return hospitals

app = Flask(__name__, static_url_path='/static', static_folder='static')
app.secret_key = 'your_secret_key_here'

app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'Soumya@2004'
app.config['MYSQL_DB'] = 'database2'

mysql = MySQL(app)

detected_symptoms = []
asked_about_wellbeing = False
predicted_disease = None
asked_for_more_symptoms = False
asked_about_info = False

def respond_to_greeting(sentence):
    global asked_about_wellbeing
    greetings = ["hey", "hello", "namaskara"]
    farewells = ["bye", "goodbye", "ok"]

    if any(word in sentence.lower() for word in greetings):
        asked_about_wellbeing = True
        return "Hello! How are you doing today?"

    elif any(word in sentence.lower() for word in farewells):
        return "Goodbye! Take care."

    else:
        return None

def respond_to_wellbeing(sentence):
    global detected_symptoms
    positive_responses = ["good", "fine", "great", "okay", "not bad"]
    negative_responses = ["not good", "unwell", "sick", "bad", "poorly", "worst", "not fine"]

    if any(word in sentence.lower() for word in negative_responses):
        return "I'm sorry to hear that. Can you describe your symptoms?"
    elif any(word in sentence.lower() for word in positive_responses):
        return "I'm glad to hear that! Take care!"
    else:
        return None

# List of common symptoms
common_symptoms = [
    'itching', 'skin_rash', 'continuous_sneezing', 'shivering', 'chills', 'joint_pain', 'stomach_pain', 'acidity',
    'ulcers_on_tongue', 'vomiting', 'fatigue', 'weight_gain', 'anxiety', 'cold_hands_and_feet', 'mood_swings',
    'weight_loss', 'restlessness', 'lethargy', 'patches_in_throat', 'cough', 'sweating', 'indigestion', 'headache',
    'nausea', 'loss_of_appetite', 'back_pain', 'constipation', 'abdominal_pain', 'mild_fever', 'yellow_urine',
    'runny_nose', 'congestion', 'chest_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'puffy_face_and_eyes',
    'brittle_nails', 'excessive_hunger', 'drying_and_tingling_lips', 'knee_pain', 'hip_joint_pain', 'muscle_weakness',
    'stiff_neck', 'swelling_joints', 'movement_stiffness', 'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side',
    'loss_of_smell', 'bladder_discomfort', 'passage_of_gases', 'internal_itching', 'depression', 'irritability',
    'muscle_pain', 'altered_sensorium', 'increased_appetite', 'visual_disturbances', 'lack_of_concentration',
    'receiving_blood_transfusion', 'receiving_unsterile_injections', 'history_of_alcohol_consumption', 'palpitations',
    'painful_walking', 'blackheads', 'scarring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
    'inflammatory_nails', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze'
]

def chatbot_response(input_sentence):
    global asked_about_wellbeing, detected_symptoms, predicted_disease, asked_for_more_symptoms, asked_about_info
    response = ""
    
    if "thanks" in input_sentence.lower() or "thank you" in input_sentence.lower():
        return "You're welcome!"

    if "symptom" in input_sentence.lower():
        symptoms_list = ", ".join(common_symptoms)
        return f"The possible symptoms are: {symptoms_list}"

    response = respond_to_greeting(input_sentence)
    if response:
        return response

    if asked_about_wellbeing:
        response = respond_to_wellbeing(input_sentence)
        if response:
            return response

        new_symptoms = extract_symptoms(input_sentence)
        detected_symptoms.extend(new_symptoms)
        detected_symptoms = list(set(detected_symptoms))

        if detected_symptoms:
            if len(detected_symptoms) < 3:
                response = "Please tell me about at least three symptoms to make a prediction."
            elif not asked_for_more_symptoms:
                response = "Can you tell me more about how you're feeling?"
                asked_for_more_symptoms = True
            elif not asked_about_info:
                predicted_disease = predict_disease(detected_symptoms)

                # Check for common symptoms
                common_detected_symptoms = [symptom for symptom in detected_symptoms if symptom in common_symptoms]
                if common_detected_symptoms:
                    response = f"The symptoms {', '.join(common_detected_symptoms)} are common and usually harmless.<br> "
                    response += f"However, along with your other symptoms, it could be {predicted_disease}.\n<br>"
                else:
                    response = f"Based on your symptoms, you might have {predicted_disease}.\n<br>"
                if 'loggedin' in session:
                    user_id = session['userid']
                    user_db = f'user_{user_id}'
                    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                    #cursor.execute(f'USE {user_db}')
                    cursor.execute(f'SELECT predicted_disease, timestamp FROM {user_db};')
                    previous_predictions = cursor.fetchall()

                matching_records = [record for record in previous_predictions if predicted_disease.lower() == record['predicted_disease'].lower()]
                
                if matching_records:
                    timestamps = "<br>".join(record['timestamp'].strftime('%Y-%m-%d %H:%M:%S') for record in matching_records)
                    response += f" You've previously had this ailment on the following dates:<br>{timestamps}<br>"


                # Fetch Wikipedia summary
                wikipedia_summary = fetch_wikipedia_summary(predicted_disease)
                if not(wikipedia_summary == 0):
                 response += f"Here is some information about {predicted_disease} from Wikipedia:<br>{wikipedia_summary}<br>"

                if 'loggedin' in session:
                    user_id = session['userid']
                    user_db = f'user_{user_id}'
                    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                    cursor.execute(f'INSERT INTO predicted_diseases (user_id, name, predicted_disease) VALUES (%s, %s, %s)', (user_id, session['name'], predicted_disease))
                    cursor.execute(f'INSERT INTO {user_db} (user_id, name, predicted_disease) VALUES (%s, %s, %s)', (user_id, session['name'], predicted_disease))
                    mysql.connection.commit()
                
                response += "Do you want information about drugs or the diseases they are prescribed for? (yes/no): "
                asked_about_info = True
                return response
            else:
                if "yes" in input_sentence.lower():
                    response = "Enter '1' to get drugs for a disease or '2' to find the disease a drug is prescribed for: "
                elif "no" in input_sentence.lower():
                    response = "Okay, take care!"
                elif input_sentence.strip() == '1':
                    response = "Please provide the name of the disease you want information about: "
                    session['awaiting_disease_input'] = True
                elif input_sentence.strip() == '2':
                    response = "Please provide the name of the drug you want information about: "
                    session['awaiting_drug_input'] = True
                else:
                    if session.get('awaiting_disease_input'):
                        drugs = suggest_drug_for_disease(df, input_sentence)
                        response = f"The drugs for {input_sentence} are: {', '.join(drugs)}"
                        session.pop('awaiting_disease_input', None)
                        response += "\nDo you want me to suggest any nearby hospitals? (Y/N): "
                        session['ask_hospital'] = True
                    elif session.get('awaiting_drug_input'):
                        diseases = find_disease_for_drug(df, input_sentence)
                        response = f"The diseases for {input_sentence} are: {', '.join(diseases)}"
                        session.pop('awaiting_drug_input', None)
                        response += "\nDo you want me to suggest any nearby hospitals? (Y/N): "
                        session['ask_hospital'] = True
                    elif session.get('ask_hospital'):
                        if input_sentence.lower() == "y":
                            response = "Please enter the pincode: "
                            session['awaiting_pincode'] = True
                            session['ask_hospital'] = False
                        else:
                            response = "Okay, take care!"
                            session.pop('ask_hospital', None)
                    elif session.get('awaiting_pincode'):
                        pincode = input_sentence.strip()
                        hospitals = extract_hospitals_by_pincode('hospitals.pdf', pincode)
                        if hospitals:
                            response = f"The hospitals in PIN code {pincode} are:\n<br>" + "\n".join(hospitals)
                        else:
                            response = f"No hospitals found in PIN code {pincode}."

                        if input_sentence.lower() == "n":
                            response = "Okay, take care!"
                        session.pop('awaiting_pincode', None)
                return response

    return response if response else "I'm sorry, I couldn't understand your input."

@app.route('/')
@app.route('/login', methods=['GET', 'POST'])
def login():
    message = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM user WHERE email = %s AND password = %s', (email, password,))
        user = cursor.fetchone()
        if user:
            session['loggedin'] = True
            session['userid'] = user['userid']
            session['name'] = user['name']
            session['email'] = user['email']
            message = 'Logged in successfully!'
            return redirect(url_for('chat'))
        else:
            message = 'Please enter correct email / password!'
    return render_template('login.html', message=message)

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('userid', None)
    session.pop('email', None)
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])

def register():

    message = ''

    if request.method == 'POST' and 'name' in request.form and 'password' in request.form and 'email' in request.form:

        userName = request.form['name']

        password = request.form['password']

        email = request.form['email']

        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)

        cursor.execute('SELECT * FROM user WHERE email = %s', (email,))

        account = cursor.fetchone()

        if account:

            message = 'Account already exists!'

        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):

            message = 'Invalid email address!'

        elif not userName or not password or not email:

            message = 'Please fill out the form!'

        else:

            cursor.execute('SELECT COUNT(*) FROM user')

            num = cursor.fetchone()['COUNT(*)']

            cursor.execute('INSERT INTO user(userid, name, email, password) VALUES (%s, %s, %s, %s)', (num + 1, userName, email, password,))

            mysql.connection.commit()



            user_db = f'user_{num + 1}'



            # Ensure the correct database is being used

            cursor.execute('USE database2')



            # Create the table with dynamic name

            create_table_query = f'''

                CREATE TABLE {user_db} (

                    id INT AUTO_INCREMENT PRIMARY KEY,

                    user_id INT,

                    Name VARCHAR(255),

                    predicted_disease VARCHAR(255),

                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP

                )

            '''

            cursor.execute(create_table_query)

            mysql.connection.commit()



            message = 'You have successfully registered!'

    elif request.method == 'POST':

        message = 'Please fill out the form!'

    return render_template('login.html', message=message)

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'loggedin' in session:
        username = session.get('name', 'Guest')  # Assuming 'name' is the key for the username in session
        if request.method == 'POST':
            user_input = request.form['user_input']
            response = chatbot_response(user_input)
            return jsonify(response=response)
        return render_template('chat1.html', username=username)
    return redirect(url_for('login'))

@app.route('/get_drugs_for_disease', methods=['POST'])
def get_drugs_for_disease():
    disease_name = request.form['disease_name']
    drugs_for_disease = suggest_drug_for_disease(df, disease_name)
    return jsonify(drugs=drugs_for_disease)

@app.route('/get_diseases_for_drug', methods=['POST'])
def get_diseases_for_drug():
    drug_name = request.form['drug_name']
    diseases_for_drug = find_disease_for_drug(df, drug_name)
    return jsonify(diseases=diseases_for_drug)

if __name__ == "__main__":
    app.run(debug=True)
