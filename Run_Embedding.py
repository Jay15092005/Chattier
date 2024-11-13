import streamlit as st
import os
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define paths to subject-specific folders
SUBJECT_FOLDER_PATHS = {
    "DSA": "maths_embeddings_file",  # Replace with your actual path
    # Add more subjects and their corresponding folder paths here
}

def load_vector_store(folder_path):
    """Load the FAISS vector store from a specified folder."""
    # Ensure we specify the names of the index and PKL files correctly
    index_file = os.path.join(folder_path, "index.faiss")
    embeddings_file = os.path.join(folder_path, "index.pkl")  # Make sure this file exists
    
    # Check if the index file exists
    if not os.path.isfile(index_file):
        raise RuntimeError(f"Index file not found: {index_file}")
        
    # Load the FAISS vector store
    return FAISS.load_local(folder_path, GoogleGenerativeAIEmbeddings(model="models/embedding-001"), allow_dangerous_deserialization=True)
def get_conversational_chain():
    
    # Define the prompt template with detailed response guidelines
    prompt_template = """You are an intelligent and thorough assistant tasked with answering questions in detail based on the provided context. Please adhere to the following guidelines:

    Comprehensive Responses: Ensure your answers are exhaustive, addressing every part of the question in detail, leaving no aspect unanswered. If the question involves multiple parts, provide separate answers for each.
    
    Structured Format: Organize your response using headings, bullet points, and numbered lists to improve readability and clarity. Separate complex ideas into distinct sections to help the user follow easily.
    
    Contextual Relevance: Base your answers strictly on the context provided. If the context does not explicitly cover the requested information, state: "The specific answer is not available in the provided context," and then provide any related or useful information that may help.
    
    Clarity and Precision: Use clear and concise language. Avoid vague or ambiguous terms, and ensure that every statement is directly related to the question. When applicable, simplify complex concepts without losing the core meaning.
    
    Examples and Analogies: Where appropriate, provide real-world examples or analogies to enhance understanding and make your explanations relatable.
    
    Fallback Information: If the context does not provide a direct answer, always offer general knowledge or insights related to the topic so that the user does not leave without an informative response.

    Context: {context}
    Question: {question}

    Answer:"""

    # Initialize the model with tuned parameters
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1, top_k=30, top_p=0.85)
    
    # Create the prompt template with the defined format
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Load the QA chain with the model and the structured prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question, vector_store):
    """Run the question against the pre-built vector store."""
    # Add "In Detail" to every word in the user question
    detailed_question = " ".join([word + "in Detail " for word in user_question.split()])
    
    docs = vector_store.similarity_search(detailed_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": detailed_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.title("Chatमित्र")

    # Allow user to select a subject
    subject_choice = st.selectbox("Choose a Subject", list(SUBJECT_FOLDER_PATHS.keys()))

    if subject_choice:
        # Load the embeddings based on the selected subject
        vector_store = load_vector_store(SUBJECT_FOLDER_PATHS[subject_choice])

        # Ask the user to input their question
        user_question = st.text_input("Ask a Question from the PDF Files")

        if user_question:
            response = user_input(user_question, vector_store)
            st.write("Reply:", response)

if __name__ == "__main__":
    main()
