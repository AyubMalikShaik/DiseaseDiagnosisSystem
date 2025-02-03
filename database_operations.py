import sqlite3
import streamlit as st

# Database Initialization
def init_db():
    conn = sqlite3.connect("app_data.db")
    cursor = conn.cursor()

    # Create Users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS Users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    # Create History table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS History (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            action TEXT NOT NULL,
            data TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    conn.close()

# Call database initialization at the start
init_db()

# User Management Functions
def register_user(username, password):
    conn = sqlite3.connect("app_data.db")
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO Users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def verify_user(username, password):
    conn = sqlite3.connect("app_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM Users WHERE username = ? AND password = ?", (username, password))
    user = cursor.fetchone()
    conn.close()
    return user is not None

# History Management Functions
import streamlit as st
@st.cache_data
def log_action(username, action, data):
    conn = sqlite3.connect("app_data.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO History (username, action, data) VALUES (?, ?, ?)", (username, action, str(data)))
    conn.commit()
    conn.close()

def get_user_history(username):
    conn = sqlite3.connect("app_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT action, data, timestamp FROM History WHERE username = ? ORDER BY timestamp DESC", (username,))
    history = cursor.fetchall()
    conn.close()
    return history

# Login, Logout, and Registration
def login():
    st.sidebar.header("Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if verify_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.sidebar.success("Logged in successfully!")
        else:
            st.sidebar.error("Invalid username or password.")

def logout():
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.sidebar.info("Logged out successfully.")
        option = st.sidebar.radio("Choose an option:", ["Login", "Register"])
        if option == "Login":
            login()
        elif option == "Register":
            register()

def register():
    st.sidebar.header("Register")
    username = st.sidebar.text_input("New Username")
    password = st.sidebar.text_input("New Password", type="password")
    if st.sidebar.button("Register"):
        if register_user(username, password):
            st.sidebar.success("User registered successfully!")
        else:
            st.sidebar.error("Username already exists.")

# Initialize Session State
# if "logged_in" not in st.session_state:
#     st.session_state.logged_in = False
#     st.session_state.username = None
# 
# # Login, Logout, and Registration Options
# if not st.session_state.logged_in:
#     option = st.sidebar.radio("Choose an option:", ["Login", "Register"])
#     if option == "Login":
#         login()
#     elif option == "Register":
#         register()
# else:
#     st.sidebar.info(f"Logged in as {st.session_state.username}")
#     logout()
# 
# # Main App Logic for Logged-in Users
# if st.session_state.logged_in:
#     st.title("🩺 Disease Prediction and Symptom Analysis")
#     
#     # Example: Log user actions (replace with actual app functionality)
#     user_input = st.text_input("Enter your input (e.g., symptoms):")
#     if user_input:
#         log_action(st.session_state.username, "Input Symptoms", user_input)
#         st.write(f"Logged input: {user_input}")
#     
#     if st.button("View History"):
#         st.write("### Your History")
#         history = get_user_history(st.session_state.username)
#         for action, data, timestamp in history:
#             st.write(f"{timestamp}: {action} - {data}")
# else:
#     st.write("Please log in to access the app.")
