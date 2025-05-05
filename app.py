import csv
import re
import multiprocessing
from multiprocessing import Pool
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import os
import time
import json
import google.generativeai as genai
from difflib import SequenceMatcher
from difflib import get_close_matches

# --- Streamlit config ---
st.set_page_config(
    page_title="🎓 Placement Info Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .centered-title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.2em;
    }
    .centered-subtitle {
        text-align: center;
        font-size: 1.2em;
        color: #555;
        margin-bottom: 1em;
    }
    </style>

    <div class="centered-title">🎓 Placement Info Assistant</div>
    <div class="centered-subtitle">
        An Intelligent Chatbot for IHRD College Placement Details
        <br>Powered by FAISS, Sentence Transformers, and Google-GenerativeAi (Gemini Flash)
    </div>
    <hr>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        color: #999;
        text-align: center;
        font-size: 0.85em;
        padding: 10px 0;
        border-top: 1px solid #eee;
        z-index: 100;
    }
    </style>
    <div class="footer">
        © 2025 Placement Info Assistant | Built with ❤️ by COLLEGE OF APPLIED SCIENCE MAVELIKKARA
    </div>
""", unsafe_allow_html=True)

# --- Configuration ---
CSV_FILE = 'placement.csv'
MEMORY_FILE = "chat_memory.json"

# --- Gemini Setup ---
genai.configure(api_key=st.secrets["api_key"])
llm_model = genai.GenerativeModel('gemini-2.0-flash')

# --- Utilities ---
def clean_field_name(field_name):
    field_name = field_name.replace('_', ' ').replace('\n', ' ').strip().capitalize()
    field_name = re.sub(' +', ' ', field_name)
    return field_name

def get_all_field_names(csv_path):
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return [clean_field_name(field) for field in reader.fieldnames]

def match_fields_from_query(query, field_names, cutoff=0.5):
    query_lower = query.lower()
    matches = [f for f in field_names if any(word in f.lower() for word in query_lower.split())]
    if not matches:
        matches = get_close_matches(query_lower, field_names, n=3, cutoff=cutoff)
    return matches

def process_row(row):
    data = {}
    institution_name = row.get('Institution Name ', '').strip()
    data["Institution Name"] = institution_name if institution_name else "Not Available"
    for field_name, field_value in row.items():
        if field_value and field_value.lower() not in ['n', 'no', 'nil']:
            clean_name = clean_field_name(field_name)
            data[clean_name] = field_value.strip()
    return data

@st.cache_resource
def load_data_and_embeddings():
    with open(CSV_FILE, 'r', encoding='utf-8') as csvfile:
        reader = list(csv.DictReader(csvfile))
        processed_data = [process_row(row) for row in reader]

    texts = [" ".join(f"{k}: {v}" for k, v in item.items()) for item in processed_data]
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return embedding_model, processed_data, texts, index


def similarity(s1, s2):
    return SequenceMatcher(None, s1, s2).ratio()

def retrieve_filtered_context(query, top_k, field_names, processed_data):
    query_lower = query.lower()
    relevant_institutions = {}
    max_similarity_score = 0.8  # Threshold for considering a close match

    for row in processed_data:
        institution_name = row.get('Institution Name', 'Unknown')
        name_lower = institution_name.lower()
        similarity_score = similarity(query_lower, name_lower)

        if similarity_score >= max_similarity_score or query_lower in name_lower or name_lower in query_lower:
            if institution_name not in relevant_institutions:
                relevant_institutions[institution_name] = {"Institution": institution_name}
                if "District" in row:
                    relevant_institutions[institution_name]["District"] = row.get("District")

            for field in field_names:
                if any(placement_keyword in field.lower() for placement_keyword in ['placement', 'recruiters', 'package', 'placed', 'contact']):
                    value = row.get(field)
                    if value and field not in relevant_institutions[institution_name]:
                        relevant_institutions[institution_name][field] = value

    filtered_context = []
    for institution_data in relevant_institutions.values():
        context_lines = [f"{key}: {value}" for key, value in institution_data.items() if value]
        filtered_context.append("\n".join(context_lines))

    # If no close match found, fall back to a more general semantic search (optional)
    if not filtered_context:
        query_emb = embedding_model.encode([query])
        index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
        texts = [" ".join(f"{k}: {v}" for k, v in item.items()) for item in processed_data]
        embeddings = embedding_model.encode(texts)
        index.add(np.array(embeddings))
        distances, indices = index.search(query_emb, min(top_k, len(processed_data)))

        for i in indices[0]:
            row = processed_data[i]
            institution_name = row.get('Institution Name', 'Unknown')
            if institution_name not in relevant_institutions:
                relevant_institutions[institution_name] = {"Institution": institution_name}
                if "District" in row:
                    relevant_institutions[institution_name]["District"] = row.get("District")
                for field in field_names:
                    if any(placement_keyword in field.lower() for placement_keyword in ['placement', 'recruiters', 'package', 'placed', 'contact']):
                        value = row.get(field)
                        if value and field not in relevant_institutions[institution_name]:
                            relevant_institutions[institution_name][field] = value
        filtered_context = []
        for institution_data in relevant_institutions.values():
            context_lines = [f"{key}: {value}" for key, value in institution_data.items() if value]
            filtered_context.append("\n".join(context_lines))

    return "\n\n".join(filtered_context)
def ask_gemini(context, question):
    history = ""
    for msg in st.session_state["messages"][-4:]:
        if msg["role"] == "user":
            history += f"\nUser: {msg['content']}"
        elif msg["role"] == "assistant":
            history += f"\nAssistant: {msg['content']}"

    prompt = f"""
You are a knowledgeable and friendly assistant specializing in placement details for IHRD colleges.
Your responses should sound natural and conversational — don't just list data, explain it briefly like a human would.
Do not tell about internal process, data and all. Act like a normal human.
Use only the provided context to generate helpful and natural responses.
If placement-related information is unavailable in the data, simply skip it.

### CONTEXT:
{context}

### CHAT HISTORY:
{history}

### USER QUESTION:
{question}

### IMPORTANT:
- Answer using only relevant fields from the context.
- Always mention the **institution name** along with placement information.
- If there are multiple colleges with similar stats, list them all clearly.
- Do not mention missing data.
- Focus **only** on placement-related details like recruiters, packages, placement rates, and contact information. Avoid mentioning courses or other non-placement details.
- Use a friendly, informative tone.

"""
    print("\n\n--- FINAL PROMPT TO GEMINI ---\n")
    print(prompt)
    print("\n--- END PROMPT ---\n")

    try:
        response = llm_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ Gemini error: {e}"

# --- Memory persistence ---
def save_memory():
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state["messages"], f)

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            st.session_state["messages"] = json.load(f)

# --- Main App Logic ---
embedding_model, processed_data, texts, index = load_data_and_embeddings()
all_field_names = get_all_field_names(CSV_FILE)
TOP_K = len(texts)

if "messages" not in st.session_state:
    st.session_state["messages"] = []
    load_memory()

if not st.session_state["messages"]:
    welcome_message = "👋 Hello! Ask me anything about IHRD college placements."
    st.session_state["messages"].append({"role": "assistant", "content": welcome_message})
    save_memory()

# --- Sidebar ---
with st.sidebar:
    st.header("🕑 Chat History")
    if st.session_state["messages"]:
        for msg in st.session_state["messages"]:
            st.markdown(f"**{msg['role'].capitalize()}**: {msg['content'][:30]}...")
    else:
        st.markdown("*No chats yet.*")
    if st.button("🧹 Clear Chat"):
        st.session_state["messages"] = []
        save_memory()
        st.rerun()
    if st.button("📥 Download Chat"):
        if st.session_state["messages"]:
            chat_text = "\n\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state["messages"]])
            st.download_button("Download as TXT", data=chat_text, file_name="chat_history.txt", mime="text/plain")

# --- Chat Display ---
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(f"<div class='chat-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

user_query = st.chat_input("Type your question about placements...")

if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(f"<div class='chat-bubble'>{user_query}</div>", unsafe_allow_html=True)
    with st.spinner("Typing..."):
        context = retrieve_filtered_context(user_query, top_k=TOP_K, field_names=all_field_names, processed_data=processed_data)
        raw_answer = ask_gemini(context, user_query)
    final_answer = ""
    with st.chat_message("assistant"):
        answer_placeholder = st.empty()
        for i in range(len(raw_answer)):
            final_answer = raw_answer[:i+1]
            answer_placeholder.markdown(f"<div class='chat-bubble'>{final_answer}</div>", unsafe_allow_html=True)
            time.sleep(0.01)
    st.session_state["messages"].append({"role": "assistant", "content": raw_answer})
    save_memory()
