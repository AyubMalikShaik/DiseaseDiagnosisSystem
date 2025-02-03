import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from itertools import combinations
import nltk
import requests
import os
import pickle
from Neighbors import buildGraph,sub_graph,visualize_3d_symptom_graph,display_strongly_connected_symptoms,display_most_freq_symptoms,buildGraphWithWeights
from difflib import SequenceMatcher
import joblib
import spacy
import pandas as pd
from fuzzywuzzy import process
from nltk.corpus import wordnet
from gensim.models import KeyedVectors
from display_communities import display_clusters
from database_operations import register_user,verify_user,log_action,get_user_history,login,logout,register
# Download necessary NLTK resources
@st.cache_data
def download_nltk_resources():
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('omw-1.4')

download_nltk_resources()
if "final_symptoms" not in st.session_state:
    st.session_state.final_symptoms = []
if "selected_symptoms" not in st.session_state:
    st.session_state.selected_symptoms = []
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.label_encoder = None
if "user_input" not in st.session_state:
    st.session_state.user_input = ""
if "community_input" not in st.session_state:
    st.session_state.community_input = ""
if "matched_symptoms" not in st.session_state:
    st.session_state.matched_symptoms =set()
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = None
# Load datasets
@st.cache_data
def load_data():
    df_comb = pd.read_csv("./Dataset/mydataset.csv")
    return df_comb
df_comb = load_data()
X = df_comb.iloc[:, 1:]
Y = df_comb.iloc[:, 0]
dataset_symptoms = list(df_comb.columns[1:])
lemmatizer = WordNetLemmatizer()
splitter = RegexpTokenizer(r'\w+')
stop_words = stopwords.words('english')
# Cache model training
@st.cache_resource
def load_model():
    mlp_model = joblib.load("./model/mlp_model.pkl")  # Use your pretrained MLP model path
    label_encoder = joblib.load("./models/label_encoder.pkl")    # Load the label encoder
    return mlp_model, label_encoder

@st.cache_data
def load_communities():
    return {1: ['shortness of breath', 'palpitations', 'sharp chest pain', 'weight gain', 'chest tightness', 'difficulty breathing', 'hurts to breath', 'dizziness', 'weakness', 'symptoms of the kidneys', 'hemoptysis', 'fatigue', 'fainting', 'feeling ill', 'recent weight loss', 'irregular heartbeat', 'increased heart rate', 'sweating', 'decreased heart rate', 'feeling cold', 'rib pain', 'burning chest pain', 'thirst', 'feeling hot'], 4: ['sharp abdominal pain', 'burning abdominal pain', 'groin mass', 'symptoms of the scrotum and testes', 'lower abdominal pain', 'regurgitation', 'upper abdominal pain', 'swollen abdomen', 'regurgitation.1', 'irregular belly button', 'vomiting', 'emotional symptoms', 'swelling of scrotum', 'long menstrual periods', 'decreased appetite', 'retention of urine', 'nausea', 'kidney mass', 'diarrhea', 'side pain', 'chills', 'vomiting blood', 'stomach bloating', 'blood in stool', 'pain of the anus', 'changes in stool appearance', 'rectal bleeding', 'constipation', 'mass in scrotum', 'melena', 'flatulence', 'hot flashes', 'pelvic pain', 'groin pain', 'jaundice', 'suprapubic pain', 'impotence', 'symptoms of bladder', 'frequent urination', 'vaginal itching', 'painful urination', 'involuntary urination', 'pain during intercourse', 'vaginal discharge', 'blood in urine', 'penis pain', 'penis redness', 'diaper rash', 'pain in testicles', 'symptoms of prostate', 'excessive urination at night', 'hesitancy', 'low urine output', 'pain during pregnancy', 'vaginal redness', 'problems during pregnancy', 'bladder mass', 'bedwetting', 'heartburn', 'cramps and spasms', 'vulvar irritation', 'heavy menstrual flow', 'vaginal pain', 'abdominal distention', 'mass or swelling around the anus', 'drainage in throat', 'flu-like syndrome', 'intermenstrual bleeding', 'loss of sex drive', 'absence of menstruation', 'vaginal bleeding after menopause', 'unpredictable menstruation', 'painful menstruation', 'infertility', 'vulvar sore', 'scanty menstrual flow', 'recent pregnancy', 'uterine contractions', 'pelvic pressure', 'penile discharge', 'itching of the anus', 'excessive growth', 'spotting or bleeding during pregnancy', 'blood clots during menstrual periods', 'frequent menstruation', 'premenstrual tension or irritability', 'flushing', 'premature ejaculation'], 2: ['arm swelling', 'back pain', 'ache all over', 'lower body pain', 'arm pain', 'neck pain', 'muscle pain', 'shoulder pain', 'shoulder stiffness or tightness', 'problems with movement', 'leg cramps or spasms', 'ankle swelling', 'leg pain', 'hip pain', 'low back pain', 'knee pain', 'elbow pain', 'paresthesia', 'stiffness all over', 'joint pain', 'hip stiffness or tightness', 'knee lump or mass', 'foot or toe pain', 'foot or toe swelling', 'loss of sensation', 'arm stiffness or tightness', 'bones are painful', 'muscle stiffness or tightness', 'ankle pain', 'hand or finger pain', 'hand or finger weakness', 'wrist pain', 'wrist swelling', 'knee swelling', 'leg swelling', 'elbow swelling', 'hand or finger swelling', 'hand or finger stiffness or tightness', 'arm weakness', 'infant feeding problem', 'knee weakness', 'back cramps or spasms', 'back stiffness or tightness', 'knee stiffness or tightness', 'leg stiffness or tightness', 'foot or toe stiffness or tightness', 'ankle weakness', 'wrist stiffness or tightness', 'foot or toe weakness', 'leg weakness', 'shoulder swelling', 'wrist lump or mass', 'skin on arm or hand looks infected', 'muscle cramps, contractures, or spasms', 'unusual color or odor to urine', 'low back cramps or spasms', 'shoulder weakness', 'bowlegged or knock-kneed', 'early or late onset of menopause', 'poor circulation', 'joint swelling', 'hand or finger cramps or spasms'], 5: ['sore throat', 'cough', 'nasal congestion', 'irritable infant', 'fever', 'coryza', 'sinus congestion', 'throat swelling', 'difficulty in swallowing', 'headache', 'throat feels tight', 'wheezing', 'pulling at ears', 'coughing up sputum', 'congestion in chest', 'peripheral edema', 'diminished hearing', 'ear pain', 'plugged feeling in ear', 'fluid in ear', 'redness in ear', 'facial pain', 'frontal headache', 'painful sinuses', 'allergic reaction', 'nosebleed', 'toothache', 'mouth ulcer', 'tongue lesions', 'mouth pain', 'swollen lymph nodes', 'dry lips', 'gum pain', 'bleeding gums', 'hoarse voice', 'ringing in ear', 'sneezing', 'abnormal breathing sounds', 'neck swelling', 'jaw swelling', 'pain in gums', 'apnea', 'neck mass', 'bleeding from ear', 'swollen or red tonsils', 'lump in throat', 'infant spitting up', 'neck stiffness or tightness', 'itchy ear(s)', 'symptoms of infants', 'throat redness', 'neck cramps or spasms'], 6: ['depressive or psychotic symptoms', 'abusing alcohol', 'anxiety and nervousness', 'depression', 'insomnia', 'abnormal involuntary movements', 'hostile behavior', 'restlessness', 'excessive anger', 'delusions or hallucinations', 'fears and phobias', 'nightmares', 'back weakness', 'drug abuse', 'smoking problems', 'low self-esteem', 'seizures', 'antisocial behavior', 'difficulty speaking', 'disturbance of memory', 'wrist weakness', 'lack of growth', 'slurring words', 'temper problems', 'focal weakness', 'obsessions and compulsions', 'difficulty eating', 'sleepiness', 'hysterical behavior', 'elbow weakness', 'muscle weakness', 'excessive appetite', 'breathing fast'], 3: ['itchy eyelid', 'diminished vision', 'double vision', 'symptoms of eye', 'pain in eye', 'foreign body sensation in eye', 'spots or clouds in vision', 'eye redness', 'blindness', 'itchiness of eye', 'swollen eye', 'cross-eyed', 'eye burns or stings', 'pus draining from ear', 'eye deviation', 'lacrimation', 'white discharge from eye', 'eyelid swelling', 'abnormal movement of eyelid', 'low back weakness', 'cloudy eye', 'mass on eyelid', 'muscle swelling', 'eye moves abnormally', 'redness in or around nose', 'bleeding from eye', 'itching of scrotum', 'eyelid lesion or rash'], 0: ['skin lesion', 'acne or pimples', 'skin growth', 'itching of skin', 'skin rash', 'skin swelling', 'abnormal appearing skin', 'irregular appearing scalp', 'skin moles', 'skin dryness, peeling, scaliness, or roughness', 'symptoms of the face', 'skin irritation', 'lip swelling', 'fluid retention', 'too little hair', 'vaginal dryness', 'lymphedema', 'irregular appearing nails', 'wrinkles on skin', 'bumps on penis', 'bleeding or discharge from nipple', 'pain or soreness of breast', 'lump or mass of breast', 'mouth dryness', 'postpartum problems of the breast', 'foot or toe lump or mass', 'skin pain', 'lip sore', 'skin on leg or foot looks infected', 'incontinence of stool', 'warts', 'back mass or lump', 'itchy scalp', 'hand or finger lump or mass', 'problems with shape or size of breast', 'unwanted hair', 'sore in nose', 'leg lump or mass', 'shoulder lump or mass', 'arm lump or mass', 'dry or flaky scalp']}
communities = load_communities()

@st.cache_data
def apply_light_theme():
    light_base_color = "2edef5"  # Light gray as the base
    highlight_color = "#507687"
    text_color = "#333333"  # Dark gray text
    button_color = "#4CAF50"
    a="#384B70"
    b="#C6E7FF"
    bg="#48A6A7"
    box_shadow = "0px 4px 8px rgba(0, 0, 0, 0.2)"  # Shadow styling
    st.markdown(
        f"""
        <style>
        /* Applying background color to the body */
        body {{
            background-color:'#48A6A7';  /* Correct background color syntax */
            color: {text_color};
            font-family: 'Book Antiqua', 'Candara', sans-serif;
        }}
         .block-container {{
            padding-left: 5rem !important;
            padding-right: 5rem !important;
            max-width: 100% !important;
        }}
        .stApp {{
        background-color: #E8F9FF !important;  /* Apply to the whole app */
    }}
        header {{
            background-color: #E8F9FF !important;
            color: #384B70 !important;
        }}
        .stButton > button {{
            background-color: {b};
            color: {highlight_color};
            border-radius: 8px;
            box-shadow: {box_shadow};
            font-family: 'Book Antiqua', 'Candara', sans-serif;
            transition: transform 0.2s ease;
        }}
        .stButton > button:hover {{
            transform: scale(1.05); /* Slight zoom effect */
        }}
        h1{{
        color:{a}
        }}
        .title-container{{
            position: fixed;
            top: 0;
            left: 0;
            z-index: 1000;

        }}
        .plotly-container {{
            background-color: #E8F9FF; /* Light blue background */
            padding: 15px;  /* Space around the chart */
            border-radius: 10px;  /* Rounded corners */
            box-shadow: 2px 4px 10px rgba(0, 0, 0, 0.1); /* Soft shadow */
            border: 2px solid #48A6A7; /* Border color */
        }}
        h3 {{
            font-family: 'Book Antiqua', 'Candara', sans-serif;
            font-size: 24px;
            color: {highlight_color};
            font-weight: bold;
        }}
        /* Styling for the text input field */
        .stTextInput input {{
            font-family: 'Book Antiqua', 'Candara', sans-serif;
            font-size: 18px;
            color: {text_color};
            padding: 10px;
            border-radius: 5px;
            border: 2px solid {highlight_color};
        }}
        
        h2{{
        color:{highlight_color};
        }}
        </style>
#     <div class="plotly-container">

        """,
        unsafe_allow_html=True
    )
st.markdown('<div class="main-container"></div>', unsafe_allow_html=True)

nlp = spacy.load("en_core_web_md")  # Use medium/large model for better accuracy
if not st.session_state.logged_in:
    option = st.sidebar.radio("Choose an option:", ["Login", "Register"])
    if option == "Login":
        login()
    elif option == "Register":
        register()
else:
    st.sidebar.info(f"Logged in as {st.session_state.username}")
    logout()

if st.session_state.logged_in:
    st.title("🩺Disease Prediction and Symptom Analysis")
    option = st.sidebar.selectbox("Choose a feature:", ["Disease Prediction", "Community Visualization","View History"])   
    if option == "Disease Prediction":
        @st.cache_data
        def get_synonyms(term):
            synonyms = set()
            for syn in wordnet.synsets(term):
                synonyms.update(syn.lemma_names())
            return list(synonyms)

        @st.cache_data
        def meaning_based_match(user_input, dataset):
            related_symptoms = []
            user_input_vec = nlp(user_input).vector

            for symptom in dataset:
                sym_vec = nlp(symptom).vector
                similarity = user_input_vec.dot(sym_vec) / (nlp(symptom).vector_norm * nlp(user_input).vector_norm)
                
                if similarity > 0.75:  # Adjust threshold as needed
                    related_symptoms.append(symptom)
            return related_symptoms

        @st.cache_data
        def fuzzy_match_symptoms(user_input, dataset, threshold=80):
            matched = process.extractBests(user_input, dataset, score_cutoff=threshold)
            return [match[0] for match in matched]
        @st.cache_data
        def extract_body_part(symptom):
            doc = nlp(symptom)
            for token in doc:
                if token.ent_type_ == "ORG" or token.pos_ in ["NOUN"]:  # Adjust based on dataset
                    return token.text.lower()
            return None
        @st.cache_data
        def find_related_symptoms(user_input):
            user_input = user_input.lower().strip()   
            related_symptoms = set()
            if user_input in dataset_symptoms:
                related_symptoms.add(user_input)# 1️⃣ Exact Match 
            synonyms = get_synonyms(user_input)
            for syn in synonyms:
                if syn in dataset_symptoms:
                    related_symptoms.add(syn)# 2️⃣ Synonym Match  
            related_symptoms.update(fuzzy_match_symptoms(user_input, dataset_symptoms))# 3️⃣ Fuzzy Matching (Fast & Accurate)  
            related_symptoms.update(meaning_based_match(user_input, dataset_symptoms))# 4️⃣ Meaning-Based Similarity Matching
            body_part = extract_body_part(user_input)# 5️⃣ Body Part Extraction & Matching
            if body_part:
                for symptom in dataset_symptoms:
                    if body_part in symptom:
                        related_symptoms.add(symptom)

            return list(related_symptoms)
        # Preprocess symptoms
        @st.cache_data
        def preprocess_symptoms(user_input):
            user_symptoms = user_input.lower().split(',')
            processed_user_symptoms = []
            for sym in user_symptoms:
                sym = sym.strip().replace('-', ' ').replace("'", '')
                sym = ' '.join([lemmatizer.lemmatize(word) for word in splitter.tokenize(sym)])
                processed_user_symptoms.append(sym)

            return processed_user_symptoms
        def update_selected_symptoms():
            st.session_state.selected_symptoms = st.session_state.temp_selected_symptoms

        def update_selected_nodes(symptom):
            key = f"selected_node_{symptom}"
            st.session_state.final_symptoms.append({
                "matched_symptom": symptom,
                "selected_node": st.session_state[key].split(",")
            })

        @st.cache_data
        def match_symptoms2(processed_symptoms):
            found_symptoms = set()
            for idx, data_sym in enumerate(dataset_symptoms):
                data_sym_split = data_sym.split()
                for user_sym in processed_symptoms:
                    count = 0
                    for symp in data_sym_split:
                        if symp in user_sym.split():
                            count += 1
                    if count / len(data_sym_split) > 0.5:
                        found_symptoms.add(data_sym)
            return list(found_symptoms)

        # Match symptoms based on similarity
        @st.cache_data
        def match_symptoms(processed_symptoms):
            found_symptoms = set()
            threshold = 0.6  # Set a similarity threshold
            for user_sym in processed_symptoms:
                for data_sym in dataset_symptoms:
                    similarity = SequenceMatcher(None, user_sym, data_sym).ratio()
                    if similarity >= threshold:
                        found_symptoms.add(data_sym)
            return list(found_symptoms)
        #Streamlit UI
        # Initialize session state if not already set
        if "model" not in st.session_state:
            st.session_state.model = None
            st.session_state.label_encoder = None
        # Apply the light theme with box shadow
        apply_light_theme()
        # Train model button
        st.header("1️⃣ Train Model")
        if st.button("Train Model"):
            if st.session_state.model is None:
                st.session_state.model, st.session_state.label_encoder = load_model()
                st.success("Model trained successfully!")
            else:
                st.warning("Model is already trained.")

        # Final symptoms list
        final_symptoms = []
        @st.cache_data
        def get_graph():
            G = buildGraph()
            G = buildGraphWithWeights()
            return G

        # Process user input for symptoms
        st.header("2️⃣ Input Symptoms")
        st.session_state.user_input = st.text_input(
                "Enter symptoms separated by commas:",
                st.session_state.user_input
            )
        if st.session_state.user_input:
            # Process the user input into a list of symptoms
            processed_user_symptoms = [
                symptom.strip().lower() for symptom in st.session_state.user_input.split(",") if symptom.strip()
            ]
            log_action(st.session_state.username, "Input Symptoms", st.session_state.user_input)
        # Find related symptoms for all processed symptoms
            matched_symptoms = set()
            for symptom in processed_user_symptoms:
                matched_symptoms.update(find_related_symptoms(symptom))  # Combine results for all input symptoms
          #     matched_symptoms = list(set(match_symptoms(processed_user_symptoms)) | set(match_symptoms2(processed_user_symptoms)))
            st.session_state.matched_symptoms = list(matched_symptoms)  # Store in session state

            if "temp_selected_symptoms" not in st.session_state:
                st.session_state.temp_selected_symptoms = st.session_state.selected_symptoms

            selected_symptoms = st.multiselect(
                "Select matching symptoms:",
                options=st.session_state.matched_symptoms,
                default=st.session_state.temp_selected_symptoms,
                key="temp_selected_symptoms",
                on_change=update_selected_symptoms
            )

            if selected_symptoms != st.session_state.selected_symptoms:
                st.session_state.selected_symptoms = selected_symptoms
                # ✅ Only update session state if selection changed

            # Build graph and visualize symptoms
            if st.session_state.selected_symptoms:
                G = get_graph()  # Only build once, cache graph if possible
                for symptom in st.session_state.selected_symptoms:
                    st.title(f"Exploring: {symptom}")
                    selected_nodes = []
                    col1,col2=st.columns(2)
                    sub=sub_graph(G,symptom)
                    with col1:
                        st.subheader(f" Most frequent occured symptoms of {symptom}")
                        fig1=display_most_freq_symptoms(G,symptom,15)
                        st.plotly_chart(fig1, use_container_width=True)
                    with col2:
                        st.subheader(f" Strongly connected symptoms to {symptom}")
                        fig2=display_strongly_connected_symptoms(G,symptom,15)
                        st.plotly_chart(fig2, use_container_width=True)

            #         html_file, selected_nodes = visualize_symptom_graph(symptom, G)
                    key = f"selected_node_{symptom}"

                    if key not in st.session_state:
                        st.session_state[key] = ""
                    sn=[]
                    sn=st.text_input(
                        "Enter the selected symptom nodes separated by commas:",
                        key=key,
                        on_change=update_selected_nodes,
                        args=(symptom,)
                    ).split(',')
                    st.write(f"Added {sn} for symptom {symptom}.")

            st.write("### Final Symptoms and Selected Nodes:")
            st.write(st.session_state.final_symptoms)

            # Disease prediction button
            st.header("3️⃣ Predict Disease")
            # Disease prediction button
            if st.button("Predict Disease"):
                if st.session_state.model is None or st.session_state.label_encoder is None:
                    st.error("Please train the model first using the Train Model button.")
                else:
                    log_action(st.session_state.username, "Prediction", st.session_state.final_symptoms)

                    # Initialize sample_x with zeros
                    sample_x = [0] * len(dataset_symptoms)
                    
                    # Populate sample_x based on symptoms
                    for sym in st.session_state.final_symptoms:
                        matched_symptom = sym["matched_symptom"]
                        if matched_symptom in dataset_symptoms:
                            sample_x[dataset_symptoms.index(matched_symptom)] = 1
                        else:
                            st.warning(f"Symptom '{matched_symptom}' not found in dataset.")
                        
                        selected_node = sym.get("selected_node", [])
                        for sn in selected_node:
                            if sn in dataset_symptoms:
                                sample_x[dataset_symptoms.index(sn)] = 1
                            else:
                                st.warning(f"Node '{sn}' not found in dataset.")
                    # Ensure sample_x is a 2D array
                    sample_x = [sample_x]
                    # Validate feature size
                    if len(sample_x[0]) != st.session_state.model.n_features_in_:
                        st.error("Feature size mismatch. Ensure the symptoms match the training dataset.")
                    else:
                        # Predict probabilities and find top predictions
                        probabilities = st.session_state.model.predict_proba(sample_x)[0]
                        topk_indices = probabilities.argsort()[-10:][::-1]
                        predicted_labels = st.session_state.label_encoder.inverse_transform(topk_indices)
                        # Display top predictions with probabilities
                        st.write("### Predicted Diseases (with Probabilities)")
                        for i, index in enumerate(topk_indices):
                            disease = predicted_labels[i]
                            prob = probabilities[index]
                            st.write(f"{i+1}. {disease} - {prob:.2%} probability")
        # Restart process button
        if st.button("Restart Process"):
            st.session_state.model = None
            st.session_state.label_encoder = None
            st.experimental_rerun()

    elif option == "Community Visualization":
        st.title("Community Graph Visualizations")
        st.session_state.community_input = st.text_input("Enter a symptom to find its community:", st.session_state.community_input)
        if st.session_state.community_input:
            input_symptom = st.session_state.community_input.lower().strip()
            matched_community = None

            for community_id, symptoms in communities.items():
                if input_symptom in symptoms:
                    matched_community = community_id
                    break
            if matched_community is not None:
                st.success(f"Your symptom belongs to Cluster {matched_community + 1}.")
                HTML_DIR = "community_htmls"
                html_path = os.path.join(HTML_DIR, f"community_{matched_community}.html")

                if os.path.exists(html_path):
                    with open(html_path, "r", encoding="utf-8") as html_file:
                        html_content = html_file.read()
                    st.components.v1.html(html_content, height=600, scrolling=True)
                else:
                    st.error(f"HTML file for Cluster {matched_community + 1} not found.")
            else:
                st.error("No matching community found for the entered symptom.")

        if st.button("Clear Input"):
            st.session_state.community_input = ""
            st.experimental_rerun()
    elif option=="View History":
        if st.button("View History"):
            st.write("### Your History")
            history = get_user_history(st.session_state.username)
            for action, data, timestamp in history:
                st.write(f"{timestamp}: {action} - {data}")

