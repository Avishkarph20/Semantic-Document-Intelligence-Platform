import streamlit as st
from PyPDF2 import PdfReader
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

GOOGLE_API_KEY = (
    os.getenv("GOOGLE_API_KEY", "").strip()
    or os.getenv("GEMINI_API_KEY", "").strip()
)
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.0-flash")
EMBEDDING_MODEL = (os.getenv("GEMINI_EMBEDDING_MODEL", "").strip() or "gemini-embedding-001")

if EMBEDDING_MODEL in {"models/text-embedding-004", "embedding-001", "models/embedding-001"}:
    EMBEDDING_MODEL = "gemini-embedding-001"

if not GOOGLE_API_KEY:
    st.warning("Google API key is not set. Add GOOGLE_API_KEY or GEMINI_API_KEY to your .env file before processing PDFs.")

def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def format_gemini_error(error):
    message = str(error).lower()
    if "resource_exhausted" in message or "429" in message or "quota" in message:
        return (
            "Google Gemini quota was exceeded, so the answer could not be generated. "
            "Please wait a bit, check your billing/quota limits, or use a different Google AI Studio account."
        )
    if "not_found" in message or "404" in message:
        return (
            "The selected Gemini model is not available for this key. "
            "Please update GEMINI_CHAT_MODEL to a supported model or use a valid Google AI Studio account."
        )
    return f"Gemini request failed: {error}"


def get_vector_store(text_chunks):
    if not GOOGLE_API_KEY:
        st.error("Google API key is missing. Please set GOOGLE_API_KEY in your .env file and restart the app.")
        return

    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY,
        )
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Failed to create embeddings: {format_gemini_error(e)}")
        return


def get_conversational_chain():
    prompt_template = """
You are a helpful assistant. Use the provided context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

    model = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY,
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return prompt | model


def user_input(user_question):
    if not GOOGLE_API_KEY:
        st.error("Google API key is missing. Please set GOOGLE_API_KEY in your .env file and restart the app.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )
    index_path = "faiss_index"

    if not os.path.exists(index_path) or not os.path.exists(os.path.join(index_path, "index.faiss")):
        st.warning("Please upload and process PDF files first so I can build the search index.")
        return

    try:
        new_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Unable to load the search index: {e}")
        return

    docs = new_db.similarity_search(user_question, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)

    try:
        chain = get_conversational_chain()
        response = chain.invoke({"context": context, "question": user_question})
    except Exception as e:
        st.error(format_gemini_error(e))
        return

    answer = getattr(response, "content", str(response))
    st.write("Reply: ", answer)




def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()