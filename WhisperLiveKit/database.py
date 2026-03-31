import sqlite3
import os
import json
import uuid
from datetime import datetime

# Define database file path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "data")
if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)

DB_PATH = os.path.join(DB_DIR, "meetings.db")

def get_db_connection():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """初始化数据库表结构"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS meetings (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            start_time TEXT NOT NULL,
            duration INTEGER NOT NULL,
            audio_path TEXT,
            transcription_data TEXT,
            created_at TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# 初始化数据库
init_db()

# CRUD 帮助函数
def get_all_meetings():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, title, start_time, duration, audio_path, created_at FROM meetings ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_meeting_by_id(meeting_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM meetings WHERE id = ?', (meeting_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        result = dict(row)
        if result['transcription_data']:
            try:
                result['transcription_data'] = json.loads(result['transcription_data'])
            except:
                pass
        return result
    return None

def create_meeting(title, start_time, duration, audio_path, transcription_data):
    meeting_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()
    
    # 确保 transcription_data 是字符串
    if not isinstance(transcription_data, str):
        transcription_data = json.dumps(transcription_data)
        
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO meetings (id, title, start_time, duration, audio_path, transcription_data, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (meeting_id, title, start_time, duration, audio_path, transcription_data, created_at))
    conn.commit()
    conn.close()
    return meeting_id

def delete_meeting(meeting_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM meetings WHERE id = ?', (meeting_id,))
    conn.commit()
    deleted = cursor.rowcount > 0
    conn.close()
    return deleted

def update_meeting_title(meeting_id, new_title):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE meetings SET title = ? WHERE id = ?', (new_title, meeting_id))
    conn.commit()
    updated = cursor.rowcount > 0
    conn.close()
    return updated
