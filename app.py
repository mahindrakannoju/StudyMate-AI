import streamlit as st
import fitz
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import google.generativeai as genai
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSMRvbzCDibay_E_opu55QrUqdZBvZHu3J5TA&s");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
""",
    unsafe_allow_html=True)
st.set_page_config(layout="wide", page_title="CONVERSATIONAL PDF Q&A CHATBOT", page_icon="📚")
st.markdown(
    """
       <style>
    .stButton>button {
        background-color: white; /* A vibrant blue for buttons */
        color: #A52A2A;
        padding: 10px 24px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 20px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
        border: none;
        transition: background-color 0.3s ease; /* Smooth transition on hover */}
    .stButton>button:hover {
        background-color: #FFFFFF; /* Darker blue on hover */}
    .stTextArea label, .stFileUploader label {
        font-size: 18px;
        font-weight: bold;
        color: #FFFFFF; /* Dark blue for labels */}
    .header {
        font-size: 32vw/vh;
        color: #FFFFFF; /* Deep blue for main headers */
        text-align: center;
        margin-bottom: 10px;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2); /* More prominent shadow */}
    .subheader {
        font-size: 32px;
        color: #FFFFFF; /* Medium blue for subheaders */
        margin-top: 10px;
        margin-bottom: 10px;
        font-weight: bold;}
    .response-box {
        background-color: #FFD700; /* Lightest blue for response box background */
        border-left: 8px solid #000000; /* Vibrant blue border */
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
        box-shadow: 1px 1px 5px rgba(196, 164, 132, 0.05); /* Very subtle shadow */}
    }
    </style>
    """,
    unsafe_allow_html=True,)
st.markdown('<p style="font-size:50px;font-weight:bold;"> <u>StudyMate:</u></p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:30px;font-weight:bold;">An AI-Powered PDF-Based Q&A System for Students </p>', unsafe_allow_html=True)
st.markdown('<p style="font-size:20px;font-weight:bold;color: #FFFFFF;text-align:center;"> ✨CONVERSATIONAL Q/A FROM ACADEMIC PDFs✨ </p>', unsafe_allow_html=True)
@st.cache_data # Cache this function to avoid re-running on every interaction
def extract_pdf_text(file_obj):
    doc = fitz.open(stream=file_obj.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text
@st.cache_data # Cache this function too
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
genai.configure(api_key="AIzaSyB0kDV0BpobT8O5dYntn5cAg3bXSW3Bq4U")
st.markdown('<p class="subheader">Upload your PDF and start asking questions!</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", help="Upload your academic PDF document here.")
if uploaded_file:
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_pdf_text(uploaded_file)
    st.success("PDF uploaded and text extracted successfully!")
    if "chunks" not in st.session_state:
        with st.spinner("Processing PDF for Q&A..."):
            chunks = chunk_text(pdf_text)
            embedder = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = embedder.encode(chunks)
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(np.array(embeddings))
            st.session_state["chunks"] = chunks
            st.session_state["embedder"] = embedder
            st.session_state["index"] = index
        st.success("PDF pre-processed and ready for questions!")
    st.markdown('<p class="subheader">Ask a question about the PDF content:</p>', unsafe_allow_html=True)
    user_question = st.text_area("Your Question:", height=100, help="Type your question related to the uploaded PDF.")
    if user_question and "embedder" in st.session_state:
        with st.spinner("Searching for relevant information and generating answer..."):
            embedder = st.session_state["embedder"]
            index = st.session_state["index"]
            chunks = st.session_state["chunks"]
            question_embedding = embedder.encode([user_question])
            D, I = index.search(np.array(question_embedding), k=3)
            retrieved_chunks = [chunks[i] for i in I[0]]
            context = "\n".join(retrieved_chunks)
            model = genai.GenerativeModel("gemini-2.5-flash")
            prompt = f"Answer the following question based on the provided context. If the answer is not in the context, state sorry the pdf has no such information.\n\nContext:\n{context}\n\nQuestion: {user_question}\n\nAnswer:"
            response = model.generate_content(prompt)
        st.markdown('<p class="subheader">Answer:</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="response-box">{response.text}</div>', unsafe_allow_html=True)
    elif user_question and "embedder" not in st.session_state:
        st.warning("Please wait for the PDF to finish processing before asking questions.")
st.markdown("---")
st.markdown('<p style="font-size:20px;font-weight:bold;color: #FFFFFF;text-align:center;">✨ GENERAL QUESTION-ANSWER CHATBOT ✨</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Ask me anything! </p>', unsafe_allow_html=True)
general_user_input = st.text_area("Type your question here:", height=100, key="general_chatbot_input")
if general_user_input:
    with st.spinner("Generating general answer..."):
        general_model = genai.GenerativeModel("gemini-2.5-flash")
        general_response = general_model.generate_content(general_user_input)
    st.markdown('<p class="subheader">General Chatbot Response:</p>', unsafe_allow_html=True)
    st.markdown(f'<div class="response-box">{general_response.text}</div>', unsafe_allow_html=True)
    st.sidebar.markdown("### About")
st.sidebar.info(
    "This application allows you to upload a PDF document and then allows you to ask questions about its content. "
    "It also includes a general-purpose chatbot for easy use."
)
st.sidebar.markdown("---")
st.sidebar.markdown("Developed with ❤ using Streamlit, FAISS, python, Sentence Transformers, and Google Gemini.")


