import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Google API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or "AIzaSyD36fJ-ckIKMEKtSE31cfToeGe0wnjhDBc"
genai.configure(api_key=GOOGLE_API_KEY)

PDF_PATH = "plant_info.pdf"
CHROMA_DB_DIR = "chroma_index"

def get_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    vector_store = Chroma.from_texts(text_chunks, embedding=embeddings, persist_directory=CHROMA_DB_DIR)
    vector_store.persist()

def get_conversational_chain():
    prompt_template = """
    You are a plant expert. Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, say "The answer is not available in the context."

    Context: {context}
    Question: {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.6,
        google_api_key=GOOGLE_API_KEY
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    new_db = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    st.write("🌱 **Plant Expert:**", response["output_text"])

def main():
    st.set_page_config("Plant AI Chat")
    st.header("🌿 Ask anything about Plants!")

    if not os.path.exists(CHROMA_DB_DIR):
        with st.spinner("📄 Reading and processing plant data..."):
            raw_text = get_pdf_text(PDF_PATH)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("✅ Plant data processed and ready to chat!")

    user_question = st.text_input("Ask a plant-related question:")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("📄 Plant PDF Info")
        st.markdown("✅ PDF automatically processed at startup.")

if __name__ == "__main__":
    main()
