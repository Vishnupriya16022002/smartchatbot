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

st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
""", unsafe_allow_html=True)

CSV_FILE = 'placement.csv'
TXT_FILE = 'institution_descriptions_placement.txt'
MEMORY_FILE = "chat_memory_placement.json"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
TOP_K = 5

try:
    genai.configure(api_key=st.secrets["api_key"])
    llm_model = genai.GenerativeModel('gemini-1.5-flash')
    gemini_configured = True
except Exception as e:
    st.error(f"💥 Failed to configure Google AI: {e}")
    st.error("Please ensure your Google API Key is correctly set in Streamlit secrets as 'api_key'. The app requires Gemini to function.")
    gemini_configured = False
    llm_model = None

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
    .chat-bubble {
        display: inline-block;
        padding: 10px 15px;
        border-radius: 18px;
        margin: 5px;
        max-width: 85%;
        word-wrap: break-word;
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


def clean_field_name(field_name):
    if not isinstance(field_name, str):
        return "Unknown Field"
    field_name = field_name.replace('_', ' ').replace('\n', ' ').strip()
    field_name = re.sub(' +', ' ', field_name)
    field_name = ' '.join(word.capitalize() for word in field_name.split())
    field_name = field_name.replace("Po ", "Placement Officer ")
    field_name = field_name.replace("Ug", "UG")
    field_name = field_name.replace("Pg", "PG")
    field_name = field_name.replace("Perc ", "% ")
    return field_name

def process_row(row):
    description = ""
    institution_name_col = 'Institution Name'
    institution_name = row.get(institution_name_col, '').strip()

    if institution_name:
        description += f"Institution: {institution_name}."
    else:
        return None

    for field_name, field_value in row.items():
        if field_name == institution_name_col:
            continue
        if field_value is None: continue
        field_value_str = str(field_value).strip()
        if not field_value_str or field_value_str.lower() in ['n', 'no', 'nil', 'na', 'n/a', 'nan']:
            continue
        clean_name = clean_field_name(field_name)
        description += f" {clean_name}: {field_value_str}."

    return description.strip()

def generate_metadata_from_csv(csv_filepath, output_txt_path):
    if os.path.exists(output_txt_path):
        print(f"'{output_txt_path}' already exists. Skipping generation.")
        st.toast(f"Using existing data index.", icon="ℹ️")
        return

    print(f"Generating descriptions from '{csv_filepath}'...")
    st.toast(f"Processing {csv_filepath} for search index...", icon="⏳")
    start_time = time.time()
    try:
        with open(csv_filepath, 'r', encoding='latin-1') as csvfile:
            content = csvfile.read()
            if content.startswith('\ufeff'):
                content = content[1:]
            reader = list(csv.DictReader(content.splitlines()))

        if not reader:
            st.error(f"CSV file '{csv_filepath}' appears to be empty or couldn't be read properly.")
            return

        paragraphs = []
        for row in reader:
            result = process_row(row)
            if result is not None:
                paragraphs.append(result)

        if not paragraphs:
            st.error(f"No valid descriptions could be generated from '{csv_filepath}'. Check the file content and 'Institution Name' column.")
            return

        with open(output_txt_path, 'w', encoding='utf-8') as outfile:
            for i, paragraph in enumerate(paragraphs):
                outfile.write(paragraph + '\n')
                if i < len(paragraphs) - 1:
                    outfile.write('-' * 40 + '\n')

        end_time = time.time()
        print(f"Finished generating descriptions to '{output_txt_path}' in {end_time - start_time:.2f} seconds.")
        st.toast(f"Data processing complete!", icon="✅")

    except UnicodeDecodeError:
        try:
            with open(csv_filepath, 'r', encoding='cp1252') as csvfile:
                content = csvfile.read()
                if content.startswith('\ufeff'):
                    content = content[1:]
                reader = list(csv.DictReader(content.splitlines()))

            if not reader:
                st.error(f"CSV file '{csv_filepath}' appears to be empty or couldn't be read properly (second attempt).")
                return

            paragraphs = []
            for row in reader:
                result = process_row(row)
                if result is not None:
                    paragraphs.append(result)

            if not paragraphs:
                st.error(f"No valid descriptions could be generated from '{csv_filepath}' (second attempt). Check the file content and 'Institution Name' column.")
                return

            with open(output_txt_path, 'w', encoding='utf-8') as outfile:
                for i, paragraph in enumerate(paragraphs):
                    outfile.write(paragraph + '\n')
                    if i < len(paragraphs) - 1:
                        outfile.write('-' * 40 + '\n')

            end_time = time.time()
            print(f"Finished generating descriptions to '{output_txt_path}' (second attempt) in {end_time - start_time:.2f} seconds.")
            st.toast(f"Data processing complete!", icon="✅")

        except Exception as e2:
            st.error(f"❌ Error processing CSV file '{csv_filepath}' with multiple encodings: {e2}")
            print(f"Error during CSV processing (alternative encoding): {e2}")
    except FileNotFoundError:
        st.error(f"❌ Error: CSV file '{csv_filepath}' not found.")
        print(f"Error: File not found at {csv_filepath}")
    except Exception as e:
        st.error(f"❌ Error processing CSV file '{csv_filepath}': {e}")
        print(f"Error during CSV processing: {e}")

@st.cache_resource(show_spinner="Loading knowledge base...")
def load_data_and_embeddings():
    if not os.path.exists(TXT_FILE):
        st.error(f"Description file '{TXT_FILE}' not found. Please ensure it was generated correctly from the CSV.")
        return None, None, None

    try:
        with open(TXT_FILE, 'r', encoding='utf-8') as file:
            content = file.read()
            texts = content.split('-' * 40)
        texts = [text.strip().replace('\n', ' ') for text in texts if text.strip()]

        if not texts:
            st.error(f"No text descriptions found in '{TXT_FILE}'. Check the generation process.")
            return None, None, None

        print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        print(f"Generating embeddings for {len(texts)} text snippets...")
        embeddings = embedding_model.encode(texts, show_progress_bar=False)

        print("Building FAISS index...")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings).astype('float32'))

        print("Knowledge base loaded successfully.")
        return embedding_model, texts, index

    except Exception as e:
        st.error(f"❌ Failed to load or build the knowledge base: {e}")
        print(f"Error during embedding/indexing: {e}")
        return None, None, None


def retrieve_relevant_context(query, embedding_model, index, texts, top_k):
    if index is None or embedding_model is None:
        return "Error: Knowledge base not loaded."
    try:
        query_emb = embedding_model.encode([query])
        distances, indices = index.search(np.array(query_emb).astype('float32'), top_k)
        valid_indices = [i for i in indices[0] if i != -1 and i < len(texts)]
        context = "\n\n".join([texts[i] for i in valid_indices])
        return context
    except Exception as e:
        print(f"Error during context retrieval: {e}")
        return f"Error retrieving context: {e}"


def ask_gemini_with_context(context, question):
    if not gemini_configured or llm_model is None:
        return "❌ Error: Gemini AI model is not configured or available."

    prompt = f"""
You are an expert AI assistant specializing in providing information about IHRD college placements based *only* on the provided context.

**Instructions:**
1. Analyze the User's Question below.
2. Carefully examine the Provided Context.
3. Answer the User's Question accurately using *only* information found within the Provided Context.
4. If the context does not contain the answer, explicitly state that the specific information is not available in the provided data. Do not invent or assume details.
5. Present the answer clearly and concisely. Use bullet points for lists if appropriate.
6. Do not mention the context itself in your final answer. Just answer the question.
7. Expand common abbreviations if possible (e.g., MEC -> Model Engineering College, if identifiable from context).

**Provided Context:**
---
{context}
---

**User's Question:**
{question}

**Answer:**
"""
    try:
        response = llm_model.generate_content(prompt)
        if response.parts:
            return response.text
        elif response.prompt_feedback.block_reason:
            return f"⚠️ Response blocked due to: {response.prompt_feedback.block_reason.name}. Try rephrasing your question."
        else:
            return "❓ Sorry, I couldn't generate a response for that query based on the available information."

    except Exception as e:
        print(f"❌ Gemini API error: {e}")
        return f"❌ An error occurred while contacting the AI model: {e}"


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
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {MEMORY_FILE}. Starting fresh.")
            st.session_state["messages"] = []
        except Exception as e:
            print(f"Error loading memory: {e}")
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
            for i, msg in enumerate(st.session_state["messages"]):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            if st.button("Clear Chat History", key="clear_history"):
                st.session_state["messages"] = []
                save_memory()
                st.rerun()

        st.markdown("---")
        st.markdown("### Download Chat History")
        if st.session_state.get("messages"):
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state["messages"]])
            b64 = base64.b64encode(history_text.encode()).decode()
            st.download_button(
                label="Download Chat",
                data=history_text,
                file_name="placement_chat_history.txt",
                mime="text/plain",
            )
        else:
            st.info("No chat history to download.")

        st.markdown("---")
        st.markdown(f"Powered by: <a href='https://ai.google.dev/gemini' target='_blank'>Gemini</a> + <a href='https://www.sbert.net/' target='_blank'>Sentence Transformers</a> + <a href='https://faiss.ai/' target='_blank'>FAISS</a>", unsafe_allow_html=True)
        st.markdown(f"Running on Streamlit Cloud in {time.tzname[0]} (UTC{time.strftime('%z')})")
