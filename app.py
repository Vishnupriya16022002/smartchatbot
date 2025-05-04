import csv
import re
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import time
import json
import google.generativeai as genai
import base64

st.set_page_config(
    page_title="🎓 Placement Info Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""<meta name="viewport" content="width=device-width, initial-scale=1.0">""", unsafe_allow_html=True)

CSV_FILE = 'placement.csv'
TXT_FILE = 'institution_descriptions_placement.txt'
MEMORY_FILE = "chat_memory_placement.json"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
TOP_K = 5

# Setup Gemini
try:
    genai.configure(api_key=st.secrets["api_key"])
    llm_model = genai.GenerativeModel('gemini-1.5-flash')
    gemini_configured = True
except Exception as e:
    st.error(f"💥 Failed to configure Google AI: {e}")
    gemini_configured = False
    llm_model = None

# Styling & Header
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
        color: #AAA;
        margin-bottom: 1em;
    }
    [data-testid="stChatMessage"] > div[data-testid="stMarkdownContainer"] {
        background-color: #262730;
        border: 1px solid #333;
        border-radius: 18px;
        padding: 10px 15px;
        float: left;
    }
    [data-testid="stChatMessage"].user-message > div[data-testid="stMarkdownContainer"] {
        background-color: #0b8e44;
        color: white;
        border-radius: 18px;
        padding: 10px 15px;
        float: right;
    }
    </style>
    <div class="centered-title">🎓 Placement Info Assistant</div>
    <div class="centered-subtitle">
        Ask questions about IHRD college placements (Powered by Gemini & Semantic Search)
        <br><span style="font-size: 0.8em; color: #888;">Enter queries like "Placement percentage at MEC" or "Companies visiting CAS Mavelikara"</span>
    </div>
    <hr style="border-color: #444;">
""", unsafe_allow_html=True)

# Text processing
def clean_field_name(field_name):
    if not isinstance(field_name, str):
        return "Unknown Field"
    field_name = field_name.replace('_', ' ').replace('\n', ' ').strip()
    field_name = re.sub(' +', ' ', field_name)
    field_name = ' '.join(word.capitalize() for word in field_name.split())
    field_name = field_name.replace("Po ", "Placement Officer ")
    field_name = field_name.replace("Ug", "UG").replace("Pg", "PG")
    field_name = field_name.replace("Perc ", "% ")
    return field_name

def process_row(row):
    description = ""
    institution_name_col = next((col for col in row if col.strip().lower() == 'institution name'), None)
    institution_name = row.get(institution_name_col, '').strip() if institution_name_col else ''
    if institution_name:
        description += f"Institution: {institution_name}."
    else:
        return None

    for field_name, field_value in row.items():
        if field_name == institution_name_col:
            continue
        if field_value is None:
            continue
        field_value_str = str(field_value).strip()
        if not field_value_str or field_value_str.lower() in ['n', 'no', 'nil', 'na', 'n/a', 'nan']:
            continue
        clean_name = clean_field_name(field_name)
        description += f" {clean_name}: {field_value_str}."
    return description.strip()

def generate_metadata_from_csv(csv_filepath, output_txt_path):
    if os.path.exists(output_txt_path):
        st.toast(f"Using existing data index.", icon="ℹ️")
        return

    st.toast(f"Processing {csv_filepath} for search index...", icon="⏳")
    start_time = time.time()

    def try_read_csv(encoding):
        with open(csv_filepath, 'r', encoding=encoding) as csvfile:
            content = csvfile.read()
            if content.startswith('\ufeff'):
                content = content[1:]
            return list(csv.DictReader(content.splitlines()))

    try:
        try:
            reader = try_read_csv('utf-8-sig')
        except UnicodeDecodeError:
            reader = try_read_csv('cp1252')

        if not reader:
            st.error(f"CSV file '{csv_filepath}' is empty or unreadable.")
            return

        normalized_rows = [{k.strip(): v for k, v in row.items()} for row in reader]
        paragraphs = [process_row(row) for row in normalized_rows if process_row(row) is not None]

        if not paragraphs:
            st.error("No valid descriptions generated. Check CSV content and column headers.")
            return

        with open(output_txt_path, 'w', encoding='utf-8') as outfile:
            for i, paragraph in enumerate(paragraphs):
                outfile.write(paragraph + '\n')
                if i < len(paragraphs) - 1:
                    outfile.write('-' * 40 + '\n')

        st.toast(f"Data processing complete!", icon="✅")

    except FileNotFoundError:
        st.error(f"❌ File not found: {csv_filepath}")
    except Exception as e:
        st.error(f"❌ Error processing CSV: {e}")

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_data_and_embeddings():
    if not os.path.exists(TXT_FILE):
        st.error(f"Missing text file '{TXT_FILE}'.")
        return None, None, None
    try:
        with open(TXT_FILE, 'r', encoding='utf-8') as file:
            texts = [text.strip().replace('\n', ' ') for text in file.read().split('-' * 40) if text.strip()]
        if not texts:
            st.error(f"No descriptions found in '{TXT_FILE}'.")
            return None, None, None

        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        embeddings = embedding_model.encode(texts, show_progress_bar=False)
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings).astype('float32'))
        return embedding_model, texts, index
    except Exception as e:
        st.error(f"❌ Failed to build index: {e}")
        return None, None, None

def retrieve_relevant_context(query, embedding_model, index, texts, top_k):
    if index is None or embedding_model is None:
        return "Error: Knowledge base not loaded."
    try:
        query_emb = embedding_model.encode([query])
        distances, indices = index.search(np.array(query_emb).astype('float32'), top_k)
        valid_indices = [i for i in indices[0] if i != -1 and i < len(texts)]
        return "\n\n".join([texts[i] for i in valid_indices])
    except Exception as e:
        return f"Error retrieving context: {e}"

def ask_gemini_with_context(context, question):
    if not gemini_configured or llm_model is None:
        return "❌ Gemini AI model not configured."

    prompt = f"""
You are an AI assistant specializing in IHRD college placements.

Use only the context below to answer the user's question.

Context:
---
{context}
---

User's Question:
{question}

Answer:"""

    try:
        response = llm_model.generate_content(prompt)
        return response.text if response.parts else "⚠️ Unable to generate a response."
    except Exception as e:
        return f"❌ Error contacting Gemini: {e}"

# Chat memory
def save_memory():
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state["messages"], f, indent=2)
    except Exception as e:
        print(f"Error saving memory: {e}")

def load_memory():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                st.session_state["messages"] = json.load(f)
        except:
            st.session_state["messages"] = []
    else:
        st.session_state["messages"] = []

if "messages" not in st.session_state:
    load_memory()

generate_metadata_from_csv(CSV_FILE, TXT_FILE)
embedding_model, texts, index = load_data_and_embeddings()

st.sidebar.header("Chat with the Assistant")
user_input = st.sidebar.text_input("Your question:", key="user_input")
chat_placeholder = st.empty()

if user_input:
    with st.spinner("Thinking..."):
        context = retrieve_relevant_context(user_input, embedding_model, index, texts, TOP_K)
        response = ask_gemini_with_context(context, user_input)
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.session_state["messages"].append({"role": "assistant", "content": response})
        save_memory()
        st.rerun()

with chat_placeholder:
    for msg in st.session_state.get("messages", []):
        st.chat_message(msg["role"]).markdown(msg["content"])

with st.sidebar:
    st.subheader("Chat History")
    if st.session_state.get("messages"):
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if st.button("Clear Chat History"):
            st.session_state["messages"] = []
            save_memory()
            st.rerun()

    st.markdown("---")
    st.markdown("### Download Chat History")
    if st.session_state.get("messages"):
        history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state["messages"]])
        st.download_button("Download Chat", data=history_text, file_name="placement_chat_history.txt", mime="text/plain")
    else:
        st.info("No chat history to download.")

    st.markdown("---")
    st.markdown("Powered by: [Gemini](https://ai.google.dev/gemini) + [Sentence Transformers](https://www.sbert.net/) + [FAISS](https://faiss.ai/)", unsafe_allow_html=True)
    st.markdown(f"Running in {time.tzname[0]} (UTC{time.strftime('%z')})")
