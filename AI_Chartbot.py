import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import PyPDF2

# ------------------ 🤖 MODEL FIX ------------------
chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

# ------------------ 🎨 ANIMATED HEADER ------------------
st.markdown("""
<style>
.marquee {
    width: 100%;
    overflow: hidden;
    white-space: nowrap;
}
.marquee span {
    display: inline-block;
    padding-left: 100%;
    animation: marquee 10s linear infinite;
    font-size: 26px;
    font-weight: bold;
    color: #00FFAA;
}
@keyframes marquee {
    0% { transform: translate(0,0); }
    100% { transform: translate(-100%,0); }
}

/* Chat bubble style */
.user-msg {
    background: #1f77b4;
    padding: 10px;
    border-radius: 10px;
    color: white;
    margin: 5px 0;
}
.bot-msg {
    background: #2ecc71;
    padding: 10px;
    border-radius: 10px;
    color: black;
    margin: 5px 0;
}
</style>

<div class="marquee">
<span>🚀 Technocrats Institute Of Technology Bhopal - AI CHATBOT 🚀</span>
</div>
""", unsafe_allow_html=True)

st.title("🤖 Smart AI Chatbot")

# ------------------ 💬 CHAT ------------------
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.history.append(("user", user_input))

    response = chatbot(user_input, max_length=100)
    reply = response[0]["generated_text"]

    st.session_state.history.append(("bot", reply))

# Display chat history
for role, msg in st.session_state.history:
    if role == "user":
        st.markdown(f'<div class="user-msg">👤 {msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-msg">🤖 {msg}</div>', unsafe_allow_html=True)

# Clear chat
if st.button("🧹 Clear Chat"):
    st.session_state.history = []

# ------------------ 📄 PDF CHAT ------------------
st.divider()
st.subheader("📄 Ask Questions from PDF")

def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def create_embeddings(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))

    return index, chunks, model

def search(query, index, chunks, model):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), k=2)
    return " ".join([chunks[i] for i in I[0]])

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    text = read_pdf(uploaded_file)
    index, chunks, model = create_embeddings(text)

    question = st.text_input("Ask question from PDF:")

    if question:
        answer = search(question, index, chunks, model)
        st.success(answer)