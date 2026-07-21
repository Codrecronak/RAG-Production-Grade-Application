from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException
from .pydantic_models import QueryInput, QueryResponse, DocumentInfo, DeleteFileRequest
from .langchain_utils import get_rag_chain
from .db_utils import insert_application_logs, get_chat_history, get_all_documents, insert_document_record, delete_document_record, delete_documents_by_session, get_db_connection
from .chroma_utils import index_document_to_chroma, delete_doc_from_chroma
import os
import uuid
import logging
logging.basicConfig(filename='app.log', level=logging.INFO)
app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id
    logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, Model: {query_input.model.value}")
    if not session_id:
        session_id = str(uuid.uuid4())

    

    chat_history = get_chat_history(session_id)
    rag_chain = get_rag_chain(query_input.model.value)
    answer = rag_chain.invoke({
        "input": query_input.question,
        "chat_history": chat_history
    })['answer']
    
    insert_application_logs(session_id, query_input.question, answer, query_input.model.value)
    logging.info(f"Session ID: {session_id}, AI Response: {answer}")
    return QueryResponse(answer=answer, session_id=session_id, model=query_input.model)

from fastapi import UploadFile, File, HTTPException
import os
import shutil

@app.post("/upload-doc")
def upload_and_index_document(file: UploadFile = File(...), session_id: str = None):
    allowed_extensions = ['.pdf', '.docx', '.html']
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed types are: {', '.join(allowed_extensions)}")
    
    temp_file_path = f"temp_{file.filename}"
    
    try:
        # Save the uploaded file to a temporary file
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_id = insert_document_record(file.filename, session_id)
        success = index_document_to_chroma(temp_file_path, file_id)
        
        if success:
            return {"message": f"File {file.filename} has been successfully uploaded and indexed.", "file_id": file_id}
        else:
            delete_document_record(file_id)
            raise HTTPException(status_code=500, detail=f"Failed to index {file.filename}.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/list-docs", response_model=list[DocumentInfo])
def list_documents():
    return get_all_documents()

@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    # Delete from Chroma
    chroma_delete_success = delete_doc_from_chroma(request.file_id)

    if chroma_delete_success:
        # If successfully deleted from Chroma, delete from our database
        db_delete_success = delete_document_record(request.file_id)
        if db_delete_success:
            return {"message": f"Successfully deleted document with file_id {request.file_id} from the system."}
        else:
            return {"error": f"Deleted from Chroma but failed to delete document with file_id {request.file_id} from the database."}
    else:
        return {"error": f"Failed to delete document with file_id {request.file_id} from Chroma."}

@app.post("/delete-session-docs")
def delete_session_documents(session_id: str):
    """Delete all documents associated with a specific session"""
    file_ids = delete_documents_by_session(session_id)
    
    if file_ids:
        for file_id in file_ids:
            delete_doc_from_chroma(file_id)
        return {"message": f"Successfully deleted {len(file_ids)} documents for session {session_id}.", "deleted_files": file_ids}
    else:
        return {"message": f"No documents found for session {session_id}."}

@app.post("/clear-all-docs")
def clear_all_documents():
    """Clear all documents from the vector store and database (for new browser sessions)"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM document_store')
        all_docs = cursor.fetchall()
        conn.close()
        
        file_ids = [doc['id'] for doc in all_docs]
        
        if file_ids:
            # Delete from Chroma
            for file_id in file_ids:
                delete_doc_from_chroma(file_id)
            
            # Delete from database
            conn = get_db_connection()
            conn.execute('DELETE FROM document_store')
            conn.commit()
            conn.close()
            
            logging.info(f"Cleared all {len(file_ids)} documents for new browser session")
            return {"message": f"Successfully cleared {len(file_ids)} documents.", "deleted_files": file_ids}
        else:
            return {"message": "No documents to clear."}
    except Exception as e:
        logging.error(f"Error clearing all documents: {str(e)}")
        return {"error": f"Failed to clear documents: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)