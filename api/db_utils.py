import sqlite3
from datetime import datetime

DB_NAME = "rag_app.db"

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_application_logs():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     session_id TEXT,
                     user_query TEXT,
                     gpt_response TEXT,
                     model TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

def insert_application_logs(session_id, user_query, gpt_response, model):
    conn = get_db_connection()
    conn.execute('INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)',
                 (session_id, user_query, gpt_response, model))
    conn.commit()
    conn.close()

def get_chat_history(session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at', (session_id,))
    messages = []
    for row in cursor.fetchall():
        messages.extend([
            {"role": "human", "content": row['user_query']},
            {"role": "ai", "content": row['gpt_response']}
        ])
    conn.close()
    return messages

def create_document_store():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if the table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='document_store'")
    table_exists = cursor.fetchone()
    
    if not table_exists:
        # Create table with session_id column
        conn.execute('''CREATE TABLE IF NOT EXISTS document_store
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         filename TEXT,
                         session_id TEXT,
                         upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    else:
        # Check if session_id column exists
        cursor.execute("PRAGMA table_info(document_store)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'session_id' not in columns:
            # Add session_id column if it doesn't exist
            try:
                conn.execute('ALTER TABLE document_store ADD COLUMN session_id TEXT')
            except Exception as e:
                print(f"Column session_id might already exist: {e}")
    
    conn.commit()
    conn.close()

def insert_document_record(filename, session_id=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO document_store (filename, session_id) VALUES (?, ?)', (filename, session_id))
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return file_id

def delete_document_record(file_id):
    conn = get_db_connection()
    conn.execute('DELETE FROM document_store WHERE id = ?', (file_id,))
    conn.commit()
    conn.close()
    return True

def delete_documents_by_session(session_id):
    """Delete all documents associated with a specific session"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id FROM document_store WHERE session_id = ?', (session_id,))
    docs = cursor.fetchall()
    
    if docs:
        conn.execute('DELETE FROM document_store WHERE session_id = ?', (session_id,))
        conn.commit()
    
    conn.close()
    return [doc['id'] for doc in docs] if docs else []

def get_all_documents():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, filename, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC')
    documents = cursor.fetchall()
    conn.close()
    return [dict(doc) for doc in documents]

# Initialize the database tables
create_application_logs()
create_document_store()
