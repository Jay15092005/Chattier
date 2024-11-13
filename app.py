import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define paths to subject-specific folders
SUBJECT_FOLDER_PATHS = {
    "Maths": "maths"
}

# Use Streamlit's session state to maintain vector store across interactions
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

if "subject_loaded" not in st.session_state:
    st.session_state["subject_loaded"] = False


def get_pdf_text(pdf_docs):
    """Extract text from PDF files."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """Split the text into manageable chunks for embeddings."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """Create FAISS vector store with GoogleGenerativeAI embeddings."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Save locally for persistence
    return vector_store


def get_conversational_chain():
    """Set up the question-answering conversational chain."""
    prompt_template = """You are an intelligent and thorough assistant tasked with answering questions in detail always based on the provided context. Please adhere to the following guidelines:

    Comprehensive Responses: Ensure your answers are exhaustive, addressing every part of the question in detail, leaving no aspect unanswered. If the question involves multiple parts, provide separate answers for each.
    Structured Format: Organize your response using headings, bullet points, and numbered lists to improve readability and clarity. Separate complex ideas into distinct sections to help the user follow easily.
    Contextual Relevance: Base your answers strictly on the context provided. If the context does not explicitly cover the requested information, state: "The specific answer is not available in the provided context," and then provide any related or useful information that may help.
    Clarity and Precision: Use clear and concise language. Avoid vague or ambiguous terms, and ensure that every statement is directly related to the question. When applicable, simplify complex concepts without losing the core meaning.
    Examples and Analogies: Where appropriate, provide real-world examples or analogies to enhance understanding and make your explanations relatable.
    Fallback Information: If the context does not provide a direct answer, always offer general knowledge or insights related to the topic so that the user does not leave without an informative response.
    Context: {context}
    Question: {question}

    Answer:"""

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    """Run the question against the pre-built vector store."""
    vector_store = st.session_state["vector_store"]
    if vector_store is None:
        st.error("Vector store not initialized. Please load a subject first.")
        return
    
    docs = vector_store.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.write("Reply:", response["output_text"])


def get_pdfs_from_folder(folder_path):
    """Get a list of PDF files from the subject folder."""
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]
    return pdf_files


def load_subject_pdfs(subject_choice):
    """Load and process PDFs based on subject choice."""
    folder_path = SUBJECT_FOLDER_PATHS[subject_choice]
    pdf_files = get_pdfs_from_folder(folder_path)

    with st.spinner("Processing PDFs..."):
        raw_text = get_pdf_text(pdf_files)
        text_chunks = get_text_chunks(raw_text)
        st.session_state["vector_store"] = get_vector_store(text_chunks)
        st.session_state["subject_loaded"] = True
        st.success(f"Processed PDFs from {subject_choice}")


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    # Allow user to select a subject
    subject_choice = st.selectbox("Choose a Subject", list(SUBJECT_FOLDER_PATHS.keys()))

    # Only process PDFs if a subject is selected and not already loaded
    if subject_choice and not st.session_state["subject_loaded"]:
        load_subject_pdfs(subject_choice)

    # Ask the user to input their question
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)


if __name__ == "__main__":
    main()
