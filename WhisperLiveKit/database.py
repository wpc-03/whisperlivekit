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
    cursor.execute('PRAGMA foreign_keys = ON')
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
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transcription_segments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            meeting_id TEXT NOT NULL,
            speaker TEXT,
            speaker_id INTEGER,
            text TEXT NOT NULL,
            start_time TEXT,
            end_time TEXT,
            sort_order INTEGER NOT NULL,
            FOREIGN KEY (meeting_id) REFERENCES meetings(id) ON DELETE CASCADE
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_segments_meeting ON transcription_segments(meeting_id)')
    conn.commit()

    # 迁移：将现有 transcription_data JSON 拆分到 transcription_segments 表
    _migrate_transcription_data(conn)
    conn.close()

def _migrate_transcription_data(conn):
    """将现有 meetings.transcription_data JSON 迁移到 transcription_segments 表"""
    cursor = conn.cursor()
    # 找出有 transcription_data 但尚未迁移的会议
    cursor.execute('''
        SELECT id, transcription_data FROM meetings
        WHERE transcription_data IS NOT NULL AND transcription_data != ''
        AND id NOT IN (SELECT DISTINCT meeting_id FROM transcription_segments)
    ''')
    rows = cursor.fetchall()
    migrated = 0
    for row in rows:
        meeting_id = row['id']
        try:
            segments = json.loads(row['transcription_data'])
            if isinstance(segments, list):
                for i, seg in enumerate(segments):
                    cursor.execute('''
                        INSERT INTO transcription_segments
                        (meeting_id, speaker, speaker_id, text, start_time, end_time, sort_order)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        meeting_id,
                        str(seg.get('speaker', '')) if seg.get('speaker') is not None else None,
                        seg.get('speaker_id'),
                        seg.get('text', ''),
                        str(seg.get('start', '')) if seg.get('start') is not None else None,
                        str(seg.get('end', '')) if seg.get('end') is not None else None,
                        i
                    ))
                migrated += 1
        except (json.JSONDecodeError, TypeError):
            continue
    if migrated > 0:
        conn.commit()


# Segment CRUD 函数
def get_segments_by_meeting(meeting_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        'SELECT * FROM transcription_segments WHERE meeting_id = ? ORDER BY sort_order',
        (meeting_id,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def create_segments_bulk(meeting_id, segments_list):
    """批量插入 segments（创建会议时用）"""
    conn = get_db_connection()
    cursor = conn.cursor()
    for i, seg in enumerate(segments_list):
        cursor.execute('''
            INSERT INTO transcription_segments
            (meeting_id, speaker, speaker_id, text, start_time, end_time, sort_order)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            meeting_id,
            str(seg.get('speaker', '')) if seg.get('speaker') is not None else None,
            seg.get('speaker_id'),
            seg.get('text', ''),
            str(seg.get('start', '')) if seg.get('start') is not None else None,
            str(seg.get('end', '')) if seg.get('end') is not None else None,
            i
        ))
    conn.commit()
    conn.close()


def update_segment_text(segment_id, new_text):
    """更新单条 segment 的文本"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE transcription_segments SET text = ? WHERE id = ?', (new_text, segment_id))
    conn.commit()
    updated = cursor.rowcount > 0
    conn.close()
    return updated


def delete_segments_by_meeting(meeting_id):
    """删除某会议的所有 segments"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM transcription_segments WHERE meeting_id = ?', (meeting_id,))
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
    if row:
        result = dict(row)
        # 从 transcription_segments 表读取，组装为兼容格式
        segments = get_segments_by_meeting(meeting_id)
        result['transcription_data'] = [
            {
                'id': seg['id'],
                'speaker': seg['speaker'],
                'speaker_id': seg['speaker_id'],
                'text': seg['text'],
                'start': seg['start_time'],
                'end': seg['end_time'],
            }
            for seg in segments
        ]
        conn.close()
        return result
    conn.close()
    return None

def create_meeting(title, start_time, duration, audio_path, transcription_data):
    meeting_id = str(uuid.uuid4())
    created_at = datetime.now().isoformat()

    # 解析 segments 列表
    if isinstance(transcription_data, str):
        segments_list = json.loads(transcription_data)
    else:
        segments_list = transcription_data

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO meetings (id, title, start_time, duration, audio_path, transcription_data, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (meeting_id, title, start_time, duration, audio_path, None, created_at))
    conn.commit()
    conn.close()

    # 批量写入 segments
    if isinstance(segments_list, list):
        create_segments_bulk(meeting_id, segments_list)

    return meeting_id

def delete_meeting(meeting_id):
    # 先删除关联的 segments
    delete_segments_by_meeting(meeting_id)
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

