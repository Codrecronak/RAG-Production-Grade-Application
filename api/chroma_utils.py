from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
import os

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)

embedding_function = GoogleGenerativeAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
)

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

def load_and_split_document(file_path: str) -> List[Document]:
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    elif file_path.endswith('.docx'):
        # Try the community Docx2txtLoader first; if it fails, fall back to python-docx
        try:
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
        except Exception:
            try:
                from docx import Document as DocxReader

                doc = DocxReader(file_path)
                text = "\n".join([p.text for p in doc.paragraphs if p.text])
                documents = [Document(page_content=text, metadata={"source": file_path})]
            except Exception as e:
                raise ValueError(f"Failed to load DOCX file {file_path}: {e}")
    elif file_path.endswith('.html'):
        loader = UnstructuredHTMLLoader(file_path)
        documents = loader.load()
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    return text_splitter.split_documents(documents)

def index_document_to_chroma(file_path: str, file_id: int) -> bool:
    try:
        splits = load_and_split_document(file_path)

        # Clean filename for citation purposes (strip temp_ prefix and path)
        clean_filename = os.path.basename(file_path)
        if clean_filename.startswith("temp_"):
            clean_filename = clean_filename[len("temp_"):]

        for split in splits:
            split.metadata['file_id'] = file_id
            split.metadata['source'] = clean_filename

        vectorstore.add_documents(splits)
        return True
    except Exception as e:
        print(f"Error indexing document: {e}")
        return False

def delete_doc_from_chroma(file_id: int):
    try:
        docs = vectorstore.get(where={"file_id": file_id})
        print(f"Found {len(docs['ids'])} document chunks for file_id {file_id}")
        
        vectorstore._collection.delete(where={"file_id": file_id})
        print(f"Deleted all documents with file_id {file_id}")
        
        return True
    except Exception as e:
        print(f"Error deleting document with file_id {file_id} from Chroma: {str(e)}")
        return False
