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

# --- Page Config ---
st.set_page_config(
    page_title="🎓 College Info Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background-color: #f0f2f6;
        color: #1c1c1e;
    }

    .dark-mode body {
        background-color: #1e1e1e !important;
        color: #f5f5f5 !important;
    }

    .stChatMessage {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 10px;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.05);
    }

    .stChatMessage.user {
        background-color: #e8f0fe;
    }

    .stChatMessage.assistant {
        background-color: #e6f4ea;
    }

    .css-18ni7ap.e8zbici2 {  /* sidebar header */
        color: #0a84ff;
    }

    .css-1v3fvcr {
        padding-top: 1rem;
    }

    </style>
""", unsafe_allow_html=True)

# --- Secrets ---
API_KEYS = [
    st.secrets["OPENROUTER_API_KEY_1"],
    st.secrets["OPENROUTER_API_KEY_2"]
]

# --- UI States ---
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False
if "api_key_index" not in st.session_state:
    st.session_state["api_key_index"] = 0
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- Memory File ---
MEMORY_FILE = "chat_memory.json"
if os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "r", encoding="utf-8") as f:
        st.session_state["messages"] = json.load(f)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.session_state["dark_mode"] = st.toggle("🌙 Dark Mode", value=st.session_state["dark_mode"])
    if st.session_state["dark_mode"]:
        st.markdown("<style>body { background-color: #1e1e1e; color: white; }</style>", unsafe_allow_html=True)

    st.header("🕑 Chat History")
    if st.session_state["messages"]:
        for m in st.session_state["messages"]:
            st.markdown(f"**{m['role'].capitalize()}**: {m['content'][:30]}...")
    else:
        st.markdown("*No chats yet.*")

    if st.button("🧹 Clear Chat"):
        st.session_state["messages"] = []
        save_memory()
        st.success("Chat history cleared!")
        st.stop()

    if st.button("📥 Download Chat"):
        chat_text = "\n\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in st.session_state["messages"]])
        st.download_button("Download as TXT", data=chat_text, file_name="chat_history.txt", mime="text/plain")

# --- File Paths ---
CSV_FILE = 'cleaned_dataset.csv'
TXT_FILE = 'institution_descriptions.txt'
EMBEDDING_FILE = "embeddings.npy"
INDEX_FILE = "faiss.index"
TEXTS_FILE = "texts.pkl"
MODEL_NAME = 'all-MiniLM-L6-v2'
OPENROUTER_MODEL = 'google/gemini-2.0-flash-exp:free'

# --- Text Cleaning ---
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

# --- Embedding Loader ---
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

# --- Context Retrieval ---
def retrieve_relevant_context(query, top_k):
    query_emb = model.encode([query])
    distances, indices = index.search(np.array(query_emb), top_k)
    return "\n\n".join([texts[i] for i in indices[0]])

# --- API Call ---
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

# --- Save Memory ---
def save_memory():
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(st.session_state["messages"], f)

# --- Init + UI ---
generate_metadata_from_csv(CSV_FILE, TXT_FILE)
model, texts, index = load_data_and_embeddings()
TOP_K = min(5, len(texts))

# --- Main UI ---
st.title("🎓 College Info Assistant")
st.markdown("##### Ask anything about colleges — accurate, fast, and friendly!")

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_query = st.chat_input("Type your question here...")

if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Thinking..."):
        context = retrieve_relevant_context(user_query, TOP_K)
        raw_answer = ask_openrouter(context, user_query)

    final_answer = ""
    with st.chat_message("assistant"):
        placeholder = st.empty()
        for i in range(len(raw_answer)):
            final_answer = raw_answer[:i+1]
            placeholder.markdown(final_answer)
            time.sleep(0.01)

    st.session_state["messages"].append({"role": "assistant", "content": raw_answer})
    save_memory()
