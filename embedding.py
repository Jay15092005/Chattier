import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_pdf_text(pdf_path):
    """Extract text from a PDF file."""
    pdf_reader = PdfReader(pdf_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split the text into manageable chunks for embeddings."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def process_pdfs_in_folder(folder_path):
    """Process all PDFs in a folder and create an embedding file."""
    all_text_chunks = []

    # Loop through all PDF files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing {pdf_path}...")
            raw_text = get_pdf_text(pdf_path)
            text_chunks = get_text_chunks(raw_text)
            all_text_chunks.extend(text_chunks)

    # Create embeddings from the combined text chunks
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(all_text_chunks, embedding=embeddings)

    # Save the FAISS index to a file with the folder name
    folder_name = os.path.basename(folder_path)
    vector_store.save_local(f"{folder_name}_embeddings_file")  # Save using folder name

    print(f"Embeddings created and saved successfully as '{folder_name}_embeddings_file'!")

if __name__ == "__main__":
    # Specify the folder containing PDF files
    folder_path = "maths"  # Replace with your folder path
    process_pdfs_in_folder(folder_path)
