import os
import re
import csv
import json
import time
import pickle
import requests
import numpy as np
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
from multiprocessing import Pool, cpu_count

# --- Streamlit Configuration ---
st.set_page_config(
    page_title="🎓 College Info Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for UI Styling ---
st.markdown("""
    <style>
    body {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f7f9fc;
    }
    .main {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 12px;
    }
    .block-container {
        padding-top: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        font-size: 16px;
        line-height: 1.6;
    }
    .chat-message.user {
        background-color: #dbeafe;
        border-left: 6px solid #3b82f6;
    }
    .chat-message.assistant {
        background-color: #ecfdf5;
        border-left: 6px solid #10b981;
    }
    .sidebar .stButton > button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# --- Session State Init ---
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False
if "api_key_index" not in st.session_state:
    st.session_state["api_key_index"] = 0
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- Secrets ---
API_KEYS = [
    st.secrets["OPENROUTER_API_KEY_1"],
    st.secrets["OPENROUTER_API_KEY_2"]
]

# --- Files and Configs ---
MEMORY_FILE = "chat_memory.json"
CSV_FILE = 'cleaned_dataset.csv'
TXT_FILE = 'institution_descriptions.txt'
EMBEDDING_FILE = "embeddings.npy"
INDEX_FILE = "faiss.index"
TEXTS_FILE = "texts.pkl"
MODEL_NAME = 'all-MiniLM-L6-v2'
OPENROUTER_MODEL = 'google/gemini-2.0-flash-exp:free'

# --- Load Memory ---
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        st.session_state["messages"] = json.load(f)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.session_state["dark_mode"] = st.toggle("🌙 Dark Mode", value=st.session_state["dark_mode"])
    
    st.markdown("---")
    st.header("🕑 Chat History")
    if st.session_state["messages"]:
        for m in st.session_state["messages"]:
            st.markdown(f"**{m['role'].capitalize()}**: {m['content'][:30]}...")
    else:
        st.markdown("*No chats yet.*")
    
    if st.button("🧹 Clear Chat"):
        st.session_state["messages"] = []
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
        st.success("Chat history cleared!")
        st.stop()

    if st.button("📥 Download Chat"):
        chat_text = "\n\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state["messages"]])
        st.download_button("Download as TXT", data=chat_text, file_name="chat_history.txt", mime="text/plain")

# --- Helper Functions ---
def clean_field_name(field_name):
    return re.sub(' +', ' ', field_name.replace('_', ' ').replace('\n', ' ').strip().capitalize())

def process_row(row):
    desc = row.get("Institution_Name", "Institution Name: Not Available").strip() + ". "
    for k, v in row.items():
        if k != "Institution_Name" and v and v.lower() not in ['n', 'no', 'not available']:
            desc += f"{clean_field_name(k)}: {v.strip()}. "
    return desc.strip()

def generate_metadata_from_csv(csv_path, output_txt):
    if os.path.exists(output_txt):
        return
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
    with Pool(processes=cpu_count()) as pool:
        processed = pool.map(process_row, reader)
    with open(output_txt, 'w', encoding='utf-8') as f:
        for p in processed:
            f.write(p + '\n' + '-' * 40 + '\n')

@st.cache_resource
def load_data_and_embeddings():
    model = SentenceTransformer(MODEL_NAME)

    if os.path.exists(EMBEDDING_FILE) and os.path.exists(INDEX_FILE) and os.path.exists(TEXTS_FILE):
        embeddings = np.load(EMBEDDING_FILE)
        index = faiss.read_index(INDEX_FILE)
        with open(TEXTS_FILE, "rb") as f:
            texts = pickle.load(f)
    else:
        with open(TXT_FILE, 'r', encoding='utf-8') as f:
            texts = [t.strip() for t in f.read().split('----------------------------------------') if t.strip()]
        embeddings = model.encode(texts, show_progress_bar=True)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings))
        np.save(EMBEDDING_FILE, embeddings)
        faiss.write_index(index, INDEX_FILE)
        with open(TEXTS_FILE, "wb") as f:
            pickle.dump(texts, f)

    return model, texts, index

def retrieve_relevant_context(query, top_k):
    query_emb = model.encode([query])
    distances, indices = index.search(np.array(query_emb), top_k)
    return "\n\n".join([texts[i] for i in indices[0]])

def ask_openrouter(context, question):
    prompt = f"""You are a helpful college assistant. Answer using the CONTEXT below. If unsure, say "I couldn't find that specific information."

    CONTEXT:
    {context}

    USER QUESTION:
    {question}

    Answer:"""

    current_index = st.session_state["api_key_index"]
    headers = {
        "Authorization": f"Bearer {API_KEYS[current_index]}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}]
    }

    try:
        res = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
        res.raise_for_status()
        data = res.json()
        if 'choices' in data:
            return data['choices'][0]['message']['content']
        elif "rate limit" in str(data).lower():
            st.session_state["api_key_index"] = (current_index + 1) % len(API_KEYS)
            return ask_openrouter(context, question)
        else:
            return f"❌ API Error: {data}"
    except Exception as e:
        if "rate limit" in str(e).lower():
            st.session_state["api_key_index"] = (current_index + 1) % len(API_KEYS)
            return ask_openrouter(context, question)
        return f"❌ Error: {e}"

# --- Run App ---
generate_metadata_from_csv(CSV_FILE, TXT_FILE)
model, texts, index = load_data_and_embeddings()
TOP_K = min(5, len(texts))

st.markdown("<h1 style='text-align: center;'>🎓 College Info Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Ask anything about colleges — accurate, fast, and friendly!</p>", unsafe_allow_html=True)

# --- Chat Display ---
for msg in st.session_state["messages"]:
    css_class = f"chat-message {msg['role']}"
    st.markdown(f'<div class="{css_class}">{msg["content"]}</div>', unsafe_allow_html=True)

# --- Input ---
user_query = st.chat_input("Type your question here...")

if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})
    st.markdown(f'<div class="chat-message user">{user_query}</div>', unsafe_allow_html=True)

    with st.spinner("Thinking..."):
        context = retrieve_relevant_context(user_query, TOP_K)
        raw_answer = ask_openrouter(context, user_query)

    st.session_state["messages"].append({"role": "assistant", "content": raw_answer})
    st.markdown(f'<div class="chat-message assistant">{raw_answer}</div>', unsafe_allow_html=True)

    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state["messages"], f)
