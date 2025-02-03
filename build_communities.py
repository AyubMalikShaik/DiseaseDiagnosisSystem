from Neighbors import buildGraph,buildGraphWithWeights,visualize_communities,find_symptom_communities
import networkx as nx
import streamlit as st
G = buildGraph()
# G = buildGraphWithWeights()
# communities = find_symptom_communities(G)
communities={1: ['shortness of breath', 'palpitations', 'sharp chest pain', 'weight gain', 'chest tightness', 'difficulty breathing', 'hurts to breath', 'dizziness', 'weakness', 'symptoms of the kidneys', 'hemoptysis', 'fatigue', 'fainting', 'feeling ill', 'recent weight loss', 'irregular heartbeat', 'increased heart rate', 'sweating', 'decreased heart rate', 'feeling cold', 'rib pain', 'burning chest pain', 'thirst', 'feeling hot'], 4: ['sharp abdominal pain', 'burning abdominal pain', 'groin mass', 'symptoms of the scrotum and testes', 'lower abdominal pain', 'regurgitation', 'upper abdominal pain', 'swollen abdomen', 'regurgitation.1', 'irregular belly button', 'vomiting', 'emotional symptoms', 'swelling of scrotum', 'long menstrual periods', 'decreased appetite', 'retention of urine', 'nausea', 'kidney mass', 'diarrhea', 'side pain', 'chills', 'vomiting blood', 'stomach bloating', 'blood in stool', 'pain of the anus', 'changes in stool appearance', 'rectal bleeding', 'constipation', 'mass in scrotum', 'melena', 'flatulence', 'hot flashes', 'pelvic pain', 'groin pain', 'jaundice', 'suprapubic pain', 'impotence', 'symptoms of bladder', 'frequent urination', 'vaginal itching', 'painful urination', 'involuntary urination', 'pain during intercourse', 'vaginal discharge', 'blood in urine', 'penis pain', 'penis redness', 'diaper rash', 'pain in testicles', 'symptoms of prostate', 'excessive urination at night', 'hesitancy', 'low urine output', 'pain during pregnancy', 'vaginal redness', 'problems during pregnancy', 'bladder mass', 'bedwetting', 'heartburn', 'cramps and spasms', 'vulvar irritation', 'heavy menstrual flow', 'vaginal pain', 'abdominal distention', 'mass or swelling around the anus', 'drainage in throat', 'flu-like syndrome', 'intermenstrual bleeding', 'loss of sex drive', 'absence of menstruation', 'vaginal bleeding after menopause', 'unpredictable menstruation', 'painful menstruation', 'infertility', 'vulvar sore', 'scanty menstrual flow', 'recent pregnancy', 'uterine contractions', 'pelvic pressure', 'penile discharge', 'itching of the anus', 'excessive growth', 'spotting or bleeding during pregnancy', 'blood clots during menstrual periods', 'frequent menstruation', 'premenstrual tension or irritability', 'flushing', 'premature ejaculation'], 2: ['arm swelling', 'back pain', 'ache all over', 'lower body pain', 'arm pain', 'neck pain', 'muscle pain', 'shoulder pain', 'shoulder stiffness or tightness', 'problems with movement', 'leg cramps or spasms', 'ankle swelling', 'leg pain', 'hip pain', 'low back pain', 'knee pain', 'elbow pain', 'paresthesia', 'stiffness all over', 'joint pain', 'hip stiffness or tightness', 'knee lump or mass', 'foot or toe pain', 'foot or toe swelling', 'loss of sensation', 'arm stiffness or tightness', 'bones are painful', 'muscle stiffness or tightness', 'ankle pain', 'hand or finger pain', 'hand or finger weakness', 'wrist pain', 'wrist swelling', 'knee swelling', 'leg swelling', 'elbow swelling', 'hand or finger swelling', 'hand or finger stiffness or tightness', 'arm weakness', 'infant feeding problem', 'knee weakness', 'back cramps or spasms', 'back stiffness or tightness', 'knee stiffness or tightness', 'leg stiffness or tightness', 'foot or toe stiffness or tightness', 'ankle weakness', 'wrist stiffness or tightness', 'foot or toe weakness', 'leg weakness', 'shoulder swelling', 'wrist lump or mass', 'skin on arm or hand looks infected', 'muscle cramps, contractures, or spasms', 'unusual color or odor to urine', 'low back cramps or spasms', 'shoulder weakness', 'bowlegged or knock-kneed', 'early or late onset of menopause', 'poor circulation', 'joint swelling', 'hand or finger cramps or spasms'], 5: ['sore throat', 'cough', 'nasal congestion', 'irritable infant', 'fever', 'coryza', 'sinus congestion', 'throat swelling', 'difficulty in swallowing', 'headache', 'throat feels tight', 'wheezing', 'pulling at ears', 'coughing up sputum', 'congestion in chest', 'peripheral edema', 'diminished hearing', 'ear pain', 'plugged feeling in ear', 'fluid in ear', 'redness in ear', 'facial pain', 'frontal headache', 'painful sinuses', 'allergic reaction', 'nosebleed', 'toothache', 'mouth ulcer', 'tongue lesions', 'mouth pain', 'swollen lymph nodes', 'dry lips', 'gum pain', 'bleeding gums', 'hoarse voice', 'ringing in ear', 'sneezing', 'abnormal breathing sounds', 'neck swelling', 'jaw swelling', 'pain in gums', 'apnea', 'neck mass', 'bleeding from ear', 'swollen or red tonsils', 'lump in throat', 'infant spitting up', 'neck stiffness or tightness', 'itchy ear(s)', 'symptoms of infants', 'throat redness', 'neck cramps or spasms'], 6: ['depressive or psychotic symptoms', 'abusing alcohol', 'anxiety and nervousness', 'depression', 'insomnia', 'abnormal involuntary movements', 'hostile behavior', 'restlessness', 'excessive anger', 'delusions or hallucinations', 'fears and phobias', 'nightmares', 'back weakness', 'drug abuse', 'smoking problems', 'low self-esteem', 'seizures', 'antisocial behavior', 'difficulty speaking', 'disturbance of memory', 'wrist weakness', 'lack of growth', 'slurring words', 'temper problems', 'focal weakness', 'obsessions and compulsions', 'difficulty eating', 'sleepiness', 'hysterical behavior', 'elbow weakness', 'muscle weakness', 'excessive appetite', 'breathing fast'], 3: ['itchy eyelid', 'diminished vision', 'double vision', 'symptoms of eye', 'pain in eye', 'foreign body sensation in eye', 'spots or clouds in vision', 'eye redness', 'blindness', 'itchiness of eye', 'swollen eye', 'cross-eyed', 'eye burns or stings', 'pus draining from ear', 'eye deviation', 'lacrimation', 'white discharge from eye', 'eyelid swelling', 'abnormal movement of eyelid', 'low back weakness', 'cloudy eye', 'mass on eyelid', 'muscle swelling', 'eye moves abnormally', 'redness in or around nose', 'bleeding from eye', 'itching of scrotum', 'eyelid lesion or rash'], 0: ['skin lesion', 'acne or pimples', 'skin growth', 'itching of skin', 'skin rash', 'skin swelling', 'abnormal appearing skin', 'irregular appearing scalp', 'skin moles', 'skin dryness, peeling, scaliness, or roughness', 'symptoms of the face', 'skin irritation', 'lip swelling', 'fluid retention', 'too little hair', 'vaginal dryness', 'lymphedema', 'irregular appearing nails', 'wrinkles on skin', 'bumps on penis', 'bleeding or discharge from nipple', 'pain or soreness of breast', 'lump or mass of breast', 'mouth dryness', 'postpartum problems of the breast', 'foot or toe lump or mass', 'skin pain', 'lip sore', 'skin on leg or foot looks infected', 'incontinence of stool', 'warts', 'back mass or lump', 'itchy scalp', 'hand or finger lump or mass', 'problems with shape or size of breast', 'unwanted hair', 'sore in nose', 'leg lump or mass', 'shoulder lump or mass', 'arm lump or mass', 'dry or flaky scalp']}
st.markdown(f"""
<style>
.block-container {{
            padding-left: 5rem !important;
            padding-right: 5rem !important;
            max-width: 100% !important;
        }}
        </style>
        """,unsafe_allow_html=True)
import os
import pickle
from Neighbors import buildGraph, find_symptom_communities, visualize_communities

# Directory for saving community graphs as HTML
HTML_DIR = "community_htmls"
os.makedirs(HTML_DIR, exist_ok=True)

# Load the graph and find communities


# Save each community graph as an HTML file
html_files = {}
for key, nodes in communities.items():
    subgraph = G.subgraph(nodes)
    fig = visualize_communities(G, subgraph)

    # Save as an HTML file
    html_path = os.path.join(HTML_DIR, f"community_{key}.html")
    fig.write_html(html_path)

    # Store the file path
    html_files[key] = html_path

# Save the HTML file mapping for later use
with open("community_html_mapping.pkl", "wb") as f:
    pickle.dump(html_files, f)

print("Community graphs saved as HTML!")
