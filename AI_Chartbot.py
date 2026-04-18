import streamlit as st
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import PyPDF2

# ------------------ 🤖 FREE MODEL ------------------
pipeline("text2text-generation", model="google/flan-t5-small")

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
</style>

<div class="marquee">
<span>🚀 Technocrats Institute Of Technology Bhopal - FREE AI CHATBOT 🚀</span>
</div>
""", unsafe_allow_html=True)

st.title("🤖 Free AI Chatbot (No API)")

# ------------------ 💬 CHAT ------------------
if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.chat_input("Ask something...")

if user_input:
    st.chat_message("user").write(user_input)

    response = chatbot(user_input, max_length=100, num_return_sequences=1)
    reply = response[0]["generated_text"]

    st.chat_message("assistant").write(reply)

# ------------------ 📄 PDF CHAT ------------------
st.divider()
st.subheader("📄 Ask from PDF")

def read_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
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