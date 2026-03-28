import os
import io
import streamlit as st
import pandas as pd
import joblib
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

# Google API Imports
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# --- SETUP & ML LOGIC ---
nltk.download('punkt', quiet=True)
model = joblib.load('spam_model.pkl')
cv = joblib.load('vectorizer.pkl')

reg = RegexpTokenizer(r'[a-z]+')
stemmer = PorterStemmer()

def clean_text_pipeline(text):
    string = str(text).lower()
    tokens = reg.tokenize(string)
    stemmed = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed)

# --- GMAIL API LOGIC ---
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def authenticate_gmail():
    creds = None
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
        
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            # --- THE CLEAN CHROME TRICK ---
            # This tells Windows to use Chrome for ANY browser request in this script
            chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe'
            if not os.path.exists(chrome_path):
                # Fallback for some systems where it's in Program Files (x86)
                chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe'
            
            # Set the environment variable so the library finds it automatically
            os.environ['BROWSER'] = chrome_path + ' %s'
            
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            
            st.info("🚀 Opening Google Login in Web Browser... Please sign in there.")
            
            # Now we use the standard call. Since we set the environment variable, 
            # it will handle the redirect_uri correctly and use Chrome!
            creds = flow.run_local_server(port=0)

        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return build('gmail', 'v1', credentials=creds)

def get_user_email(service):
    """Fetches the email address of the authenticated user."""
    try:
        profile = service.users().getProfile(userId='me').execute()
        return profile.get('emailAddress')
    except Exception:
        return "Unknown User"

def fetch_emails(service, max_results=20):
    results = service.users().messages().list(userId='me', maxResults=max_results).execute()
    messages = results.get('messages', [])
    email_data = []

    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
        payload = msg_data.get('payload', {})
        headers = payload.get('headers', [])
        
        subject = "No Subject"
        sender = "Unknown"
        for header in headers:
            if header['name'] == 'Subject':
                subject = header['value']
            if header['name'] == 'From':
                sender = header['value']
                
        snippet = msg_data.get('snippet', '')
        email_data.append({'Sender': sender, 'Subject': subject, 'Content': snippet})
        
    return pd.DataFrame(email_data)

# --- STREAMLIT UI ---
st.set_page_config(page_title="Live Email Classifier", layout="wide")
st.title("📧 Live Gmail Spam Classifier Dashboard")
st.write("Minor Project - Developed by J Yash Rao and Jeet Bheriya")

# --- SIDEBAR: ACCOUNT INFO ---
st.sidebar.title("Account Settings")
if os.path.exists('token.json'):
    try:
        # Silently authenticate to show the email in sidebar
        temp_service = authenticate_gmail()
        current_user = get_user_email(temp_service)
        st.sidebar.success(f"**Logged in as:**\n{current_user}")
        
        if st.sidebar.button("Logout / Switch Account"):
            os.remove('token.json')
            st.rerun()
    except Exception:
        st.sidebar.error("Session expired or invalid.")
        if st.sidebar.button("Reconnect"):
            os.remove('token.json')
            st.rerun()
else:
    st.sidebar.warning("No account connected.")

st.markdown("---")

num_emails = st.slider("How many recent emails do you want to fetch from your inbox?", 5, 100, 20)

if st.button("Fetch and Classify My Live Emails"):
    with st.spinner("Accessing Gmail and running AI Classification..."):
        try:
            # 1. Connect to Gmail
            service = authenticate_gmail()
            
            # 2. Fetch Emails
            df = fetch_emails(service, max_results=num_emails)
            st.info(f"Successfully fetched {len(df)} emails.")
            
            # 3. Clean Text and Predict
            cleaned_content = df['Content'].apply(clean_text_pipeline)
            vectorized_content = cv.transform(cleaned_content).toarray()
            predictions = model.predict(vectorized_content)
            
            # 4. Hybrid Labeling Logic (AI + Keywords)
            df['pred_code'] = predictions 
            
            def get_final_label(row):
                if row['pred_code'] == 1:
                    return "🚨 SPAM"
                
                text_to_check = (str(row['Subject']) + " " + str(row['Content'])).lower()
                
                if any(word in text_to_check for word in ['job', 'hiring', 'career', 'interview', 'vacancy', 'offer', 'internship']):
                    return "💼 JOB"
                elif any(word in text_to_check for word in ['bank', 'otp', 'transaction', 'payment', 'invoice', 'statement', 'bill']):
                    return "💳 FINANCE"
                elif any(word in text_to_check for word in ['linkedin', 'social', 'follow', 'friend', 'newsletter', 'youtube', 'facebook']):
                    return "📱 SOCIAL"
                else:
                    return "✅ SAFE"

            df['AI Prediction'] = df.apply(get_final_label, axis=1)
            df = df[['AI Prediction', 'Sender', 'Subject', 'Content']]
            
            st.success("Classification Complete!")
            st.dataframe(df, use_container_width=True)
            
            # 5. Excel Download
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Classified Emails')
            
            st.download_button(
                label="📥 Download Classified Emails as Excel",
                data=buffer,
                file_name="My_Classified_Inbox.xlsx",
                mime="application/vnd.ms-excel"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write("Ensure your credentials.json file is correctly placed in the folder.")