import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from itertools import combinations
import nltk
import requests
from bs4 import BeautifulSoup
from Neighbors import buildGraph, suggest_related_symptoms_and_graph
from Visualize_graph import visualize_symptom_graph
from difflib import SequenceMatcher

# Download necessary NLTK resources
@st.cache_data
def download_nltk_resources():
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('omw-1.4')

download_nltk_resources()


# Load datasets
@st.cache_data
def load_data():
    df_comb = pd.read_csv("./Dataset/dis_sym_dataset_comb.csv")
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
def train_model():
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y.values.ravel())
    xgb_model = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False)
    xgb_model.fit(X, Y_encoded)
    return xgb_model, label_encoder

# Cache synonym fetching
@st.cache_data
def get_synonyms(term):
    synonyms = set()
    try:
        response = requests.get(f'https://www.thesaurus.com/browse/{term}')
        soup = BeautifulSoup(response.content, "html.parser")
        container = soup.find('section', {'class': 'MainContentContainer'})
        row = container.find('div', {'class': 'css-191l5o0-ClassicContentCard'}).find_all('li')
        synonyms = {x.get_text() for x in row}
    except:
        pass
    for syn in wordnet.synsets(term):
        synonyms.update(syn.lemma_names())
    return synonyms

# Preprocess symptoms
@st.cache_data
def preprocess_symptoms(user_input):
    user_symptoms = user_input.lower().split(',')
    processed_user_symptoms = []
    for sym in user_symptoms:
        sym = sym.strip().replace('-', ' ').replace("'", '')
        sym = ' '.join([lemmatizer.lemmatize(word) for word in splitter.tokenize(sym)])
        processed_user_symptoms.append(sym)

    user_symptoms = []
    for user_sym in processed_user_symptoms:
        user_sym = user_sym.split()
        str_sym = set()
        for comb in range(1, len(user_sym)+1):
            for subset in combinations(user_sym, comb):
                subset = ' '.join(subset)
                subset_synonyms = get_synonyms(subset)
                str_sym.update(subset_synonyms)
        str_sym.add(' '.join(user_sym))
        user_symptoms.append(' '.join(str_sym).replace('_', ' '))

    return user_symptoms

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

  
# Streamlit UI
st.title("Disease Prediction and Symptom Analysis")

# Light Theme Styling
@st.cache_data
def apply_light_theme():
    light_base_color = "#f5f5f5"  # Light gray as the base
    highlight_color = "#4CAF50"  # Soft green
    text_color = "#333333"  # Dark gray text
    button_color = "#4CAF50"
    st.markdown(
        f"""
        <style>
        body {{
            background-color: {light_base_color};
            color: {text_color};
        }}
        .stButton > button {{
            background-color: {button_color};
            color: white;
            border-radius: 8px;
        }}
        .stTextInput > div {{
            border-radius: 8px;
            background-color: white;
            color: {text_color};
        }}
        .stMultiSelect > div {{
            background-color: white;
            color: {text_color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply the light theme
apply_light_theme()

# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.label_encoder = None

# Train model button
if st.button("Train Model"):
    if st.session_state.model is None:
        st.session_state.model, st.session_state.label_encoder = train_model()
        st.success("Model trained successfully!")
    else:
        st.warning("Model is already trained.")

# Symptom matching and prediction logic
user_input = st.text_input("Enter symptoms separated by commas:")
if user_input:
    processed_user_symptoms = preprocess_symptoms(user_input)
    matched_symptoms = list(set(match_symptoms(processed_user_symptoms))|set(match_symptoms2(processed_user_symptoms)))
    
    selected_symptoms = st.multiselect("Select matching symptoms:", matched_symptoms)

    if st.button("Predict Disease"):
        if st.session_state.model is None or st.session_state.label_encoder is None:
            st.error("Please train the model first using the Train Model button.")
        else:
            sample_x = [0] * len(dataset_symptoms)
            for sym in selected_symptoms:
                if sym in dataset_symptoms:
                    sample_x[dataset_symptoms.index(sym)] = 1
            
            prediction = st.session_state.model.predict_proba([sample_x])
            topk = prediction[0].argsort()[-10:][::-1]
            predicted_labels = st.session_state.label_encoder.inverse_transform(topk)
            st.write("### Top Predicted Diseases")
            for idx, disease in enumerate(predicted_labels):
                prob = prediction[0][topk[idx]] * 100
                st.write(f"{idx + 1}. {disease}: {round(prob, 2)}%")
