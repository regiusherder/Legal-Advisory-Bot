from PyPDF2 import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import os
import warnings
warnings.filterwarnings("ignore")
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
llm = GoogleGenerativeAI(model="models/gemini-1.0-pro", google_api_key=os.getenv("GOOGLE_API_KEY"))
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1024,
    chunk_overlap=128
)

def get_text_from_pdf(pdf_path):
    # All pdfs in the data folder
    pdf_files = [f for f in os.listdir(pdf_path) if f.endswith(".pdf")]
    text = ""
    for pdf_file in pdf_files:
        with open(f"{pdf_path}/{pdf_file}", "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text += page.extract_text()
    return text

def get_text_from_pdf_chunks(pdf_path):
    text = get_text_from_pdf(pdf_path)
    chunks = text_splitter.split_text(text)
    return chunks

def store_embeddings(pdf_path):
    chunks = get_text_from_pdf_chunks(pdf_path)
    docs = FAISS.from_texts(chunks, embeddings)
    print(docs)
    docs.save_local("vectors/faiss_index")

if not os.path.exists("vectors/faiss_index"):
    store_embeddings("data/")

while True:
    chain = load_qa_chain(llm, chain_type="stuff")
    query = input("Ask me anything: ")
    if query == "exit":
        break
    loaded_embeddings = FAISS.load_local(embeddings=embeddings, folder_path="vectors/faiss_index", allow_dangerous_deserialization=True)
    relevant_docs = loaded_embeddings.similarity_search(query)
    print(chain.run(input_documents=relevant_docs, question=query))