import streamlit as st
import os
import requests
import re
import time
from difflib import get_close_matches

st.set_page_config(page_title="🎓 Kerala College Chatbot", layout="wide")

# 🔐 Load API keys from Streamlit secrets (for Streamlit Cloud)
PRIMARY_GROQ_KEY = st.secrets.get("GROQ_API_KEY", "")
BACKUP_KEYS = st.secrets.get("GROQ_BACKUP_API_KEYS", "").split(",")
ALL_GROQ_KEYS = [PRIMARY_GROQ_KEY] + [k.strip() for k in BACKUP_KEYS if k.strip()]

STOPWORDS = {
    "the", "in", "of", "with", "is", "a", "an", "colleges", "college", "for",
    "show", "and", "which", "give", "list", "that", "on", "at", "to", "by", "as"
}

def normalize_text(text):
    text = text.lower()
    abbreviations = {
        "cas": "college of applied science",
        "ce": "college of engineering",
        "thss": "technical higher secondary school"
    }
    for abbr, full in abbreviations.items():
        text = text.replace(abbr, full)
    return text

def extract_keywords(text):
    text = normalize_text(text)
    words = re.findall(r"\b\w+\b", text)
    return set(w for w in words if w not in STOPWORDS)

@st.cache_data
def load_colleges():
    with open("institution_descriptions.txt", "r", encoding="utf-8") as f:
        raw = f.read()
    blocks = raw.split("----------------------------------------")
    colleges = []
    for block in blocks:
        if not block.strip():
            continue
        data = {
            "name": "",
            "location": "",
            "courses": [],
            "type": "",
            "text": block.strip(),
            "keywords": extract_keywords(block),
            "category": "other"
        }
        name_match = re.search(r"^(.*?)\.\s*Located in (.*?).", block, re.IGNORECASE)
        if name_match:
            data["name"] = name_match.group(1).strip()
            data["location"] = name_match.group(2).strip()
        type_match = re.search(r"Belongs to the group of (.*?)\.", block)
        if type_match:
            data["type"] = type_match.group(1).strip()
        course_matches = re.findall(r"Undergraduate Courses: (.*?)\.|Postgraduate Courses: (.*?)\.", block)
        all_courses = []
        for ug, pg in course_matches:
            all_courses += [c.strip() for c in (ug + " " + pg).split(",") if c.strip()]
        data["courses"] = all_courses

        # Improved category tagging
        text_lower = data["text"].lower()
        name_lower = data["name"].lower()

        if (
            "engineering" in name_lower
            or any("btech" in c.lower() for c in data["courses"])
            or "ktu" in text_lower
        ):
            data["keywords"].add("engineering")
            data["category"] = "engineering"
        elif "applied science" in name_lower:
            data["keywords"].update(["applied", "science"])
            data["category"] = "applied science"
        elif "thss" in name_lower or "technical higher secondary" in name_lower:
            data["keywords"].add("thss")
            data["category"] = "thss"

        colleges.append(data)
    return colleges

def match_colleges(user_query, colleges):
    query_keywords = extract_keywords(user_query)
    matches = []
    for college in colleges:
        score = len(query_keywords & college["keywords"])
        if score > 0:
            if "engineering" in query_keywords and college["category"] == "engineering":
                score += 2
            elif "applied" in query_keywords and "science" in query_keywords and college["category"] == "applied science":
                score += 2
        if score > 0:
            matches.append((score, college))
    matches.sort(reverse=True, key=lambda x: x[0])
    return [m[1] for m in matches]

def get_closest_blocks(query, colleges, n=3):
    all_texts = [c["text"] for c in colleges]
    close = get_close_matches(normalize_text(query), all_texts, n=n)
    return [c for c in colleges if c["text"] in close]

def chunk_college_data(matches, max_chars=3000):
    chunks, current = [], ""
    for c in matches:
        entry = c["text"].strip() + "\n\n"
        if len(current) + len(entry) < max_chars:
            current += entry
        else:
            chunks.append(current)
            current = entry
    if current:
        chunks.append(current)
    return chunks[:5]

def call_llama(query, context, max_retries=3):
    url = "https://api.groq.com/openai/v1/chat/completions"
    system_prompt = (
        "You are a helpful assistant for students asking about colleges in Kerala. "
        "Only answer using the provided context. Do not guess or add anything outside the data."
    )
    user_prompt = f"User query: {query}\n\nMatching college info:\n{context}\n\nNow provide the best answer."

    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.5
    }

    for groq_key in ALL_GROQ_KEYS:
        headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
        for attempt in range(max_retries):
            try:
                if not st.session_state.get("shown_llama_wait_message", False):
                    st.info("⏳ Just a moment while I fetch your result...")
                    st.session_state["shown_llama_wait_message"] = True

                res = requests.post(url, headers=headers, json=payload)
                data = res.json()

                if 'choices' in data:
                    return data['choices'][0]['message']['content']
                elif 'error' in data and "Rate limit" in data['error'].get('message', ''):
                    time.sleep(10 + attempt * 5)
            except Exception:
                time.sleep(5)
    return "❌ All Groq keys failed due to rate limits or errors. Please try again later."

def generate_answer_llama(query, matches):
    st.session_state["shown_llama_wait_message"] = False

    # Filter by category if present in query
    category_filter = None
    q = query.lower()
    if "engineering" in q:
        category_filter = "engineering"
    elif "applied science" in q:
        category_filter = "applied science"
    elif "thss" in q or "technical higher secondary" in q:
        category_filter = "thss"

    if category_filter:
        filtered = [c for c in matches if c["category"] == category_filter]
        if filtered:
            matches = filtered

    if not matches:
        return "❌ Sorry, I couldn't find any matching college in the dataset."

    if len(matches) == 1:
        return call_llama(query, matches[0]["text"])

    chunks = chunk_college_data(matches, max_chars=3000)
    partials = [call_llama(query, chunk).strip() for chunk in chunks]
    combined = "\n\n".join(partials)

    summary_prompt = (
        f"You are a helpful assistant summarizing college information for a student.\n\n"
        f"Context:\n{combined}\n\n"
        f"User query: {query}\n\n"
        "✅ Summarize each college clearly one per line using this format:\n"
        "• [College Name], [District] - Offers [courses].\n"
        "👉 Use proper line breaks. Each college must be on its own line.\n"
        "📌 Do not add colleges not found in the context above.\n"
    )

    return call_llama(query, summary_prompt)

# App state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "colleges" not in st.session_state:
    st.session_state.colleges = load_colleges()

# UI
st.markdown("<h1 style='text-align:center;'>🎓 Kerala College Info Chatbot</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Ask about colleges, courses, or principal info in Kerala.</p>", unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("💬 Ask your question:", placeholder="e.g., list engineering colleges")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    matches = match_colleges(user_input, st.session_state.colleges)
    if not matches:
        matches = get_closest_blocks(user_input, st.session_state.colleges)
    with st.spinner("🧠 Generating answer..."):
        response = generate_answer_llama(user_input, matches)
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", response))

# Chat display
st.markdown("### 💬 Conversation")
for sender, msg in st.session_state.chat_history:
    color = "#64b5f6" if sender == "user" else "#2196f3"
    role = "🧑 You" if sender == "user" else "🤖 Bot"
    st.markdown(
        f"""<div style='margin-bottom:12px;padding:12px;
                    background-color: rgba(33, 150, 243, 0.1);
                    border-left: 4px solid {color};
                    border-radius: 8px;
                    font-size: 16px;'>
            <strong>{role}:</strong><br>{msg}</div>""",
        unsafe_allow_html=True
    )

# Utilities
col1, col2 = st.columns(2)
with col1:
    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []

with col2:
    if st.download_button("💾 Download Chat",
        data="\n\n".join(
            f"You: {q[1]}\nBot: {a[1]}" for q, a in zip(
                st.session_state.chat_history[::2], st.session_state.chat_history[1::2]
            )
        ),
        file_name="chat_history.txt", mime="text/plain"):
        st.success("✅ Chat downloaded!")
