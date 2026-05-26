import sqlite3
import os
import json
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
from utils.config import SQLITE_CONFIG

logger = logging.getLogger(__name__)

DEFAULT_USER_ID = "local_user"

class DatabaseService:
    def __init__(self):
        self.db_path = SQLITE_CONFIG["database_path"]
        self.check_same_thread = SQLITE_CONFIG["check_same_thread"]
        self._ensure_database_directory()
        self._initialize_database()

    def _ensure_database_directory(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
            logger.info(f"Created database directory: {db_dir}")

    def _get_connection(self):
        return sqlite3.connect(
            self.db_path,
            check_same_thread=self.check_same_thread
        )

    def _initialize_database(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS arxiv_papers (
                    arxiv_id TEXT PRIMARY KEY,
                    title TEXT,
                    authors TEXT,
                    abstract TEXT,
                    categories TEXT,
                    published_date DATE,
                    url TEXT,
                    embedding_id TEXT,
                    embedding_model TEXT,
                    embedded_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_liked_papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL DEFAULT 'local_user',
                    arxiv_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, arxiv_id),
                    FOREIGN KEY(arxiv_id) REFERENCES arxiv_papers(arxiv_id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_disliked_papers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL DEFAULT 'local_user',
                    arxiv_id TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, arxiv_id),
                    FOREIGN KEY(arxiv_id) REFERENCES arxiv_papers(arxiv_id)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS user_interest_vectors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL UNIQUE,
                    vector_data TEXT NOT NULL,
                    paper_count INTEGER NOT NULL,
                    embedding_model TEXT NOT NULL,
                    vector_dimension INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS paper_qa_index (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    arxiv_id TEXT NOT NULL UNIQUE,
                    collection_name TEXT,
                    status TEXT DEFAULT 'not_indexed',
                    chunk_count INTEGER DEFAULT 0,
                    embedding_model TEXT,
                    pdf_path TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            logger.info("Database tables initialized successfully")

    def add_liked_paper(self, user_id: str = DEFAULT_USER_ID, arxiv_id: str = None) -> bool:
        try:
            if arxiv_id is None:
                logger.error("arxiv_id is required")
                return False
                
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM user_disliked_papers WHERE user_id = ? AND arxiv_id = ?
                ''', (user_id, arxiv_id))
                
                cursor.execute('''
                    INSERT OR IGNORE INTO user_liked_papers (user_id, arxiv_id)
                    VALUES (?, ?)
                ''', (user_id, arxiv_id))
                
                conn.commit()
                logger.info(f"Paper {arxiv_id} added to liked list for user: {user_id}")
                return True
        except Exception as e:
            logger.error(f"Error adding liked paper: {str(e)}")
            return False

    def remove_liked_paper(self, user_id: str = DEFAULT_USER_ID, arxiv_id: str = None) -> bool:
        try:
            if arxiv_id is None:
                logger.error("arxiv_id is required")
                return False
                
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM user_liked_papers WHERE user_id = ? AND arxiv_id = ?
                ''', (user_id, arxiv_id))
                
                conn.commit()
                logger.info(f"Paper {arxiv_id} removed from liked list for user: {user_id}")
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error removing liked paper: {str(e)}")
            return False

    def get_liked_papers(self, user_id: str = DEFAULT_USER_ID) -> List[str]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT arxiv_id FROM user_liked_papers WHERE user_id = ? ORDER BY created_at DESC
                ''', (user_id,))
                
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting liked papers: {str(e)}")
            return []

    def get_liked_papers_with_details(self, user_id: str = DEFAULT_USER_ID) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT p.arxiv_id, p.title, p.abstract, p.authors, p.categories, p.published_date
                    FROM user_liked_papers ulp
                    JOIN arxiv_papers p ON ulp.arxiv_id = p.arxiv_id
                    WHERE ulp.user_id = ?
                    ORDER BY ulp.created_at DESC
                ''', (user_id,))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'arxiv_id': row[0],
                        'title': row[1],
                        'abstract': row[2],
                        'authors': row[3],
                        'categories': row[4],
                        'published_date': row[5]
                    })
                return results
        except Exception as e:
            logger.error(f"Error getting liked papers with details: {str(e)}")
            return []

    def is_liked_paper(self, user_id: str = DEFAULT_USER_ID, arxiv_id: str = None) -> bool:
        try:
            if arxiv_id is None:
                return False
                
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*) FROM user_liked_papers WHERE user_id = ? AND arxiv_id = ?
                ''', (user_id, arxiv_id))
                
                return cursor.fetchone()[0] > 0
        except Exception as e:
            logger.error(f"Error checking liked paper: {str(e)}")
            return False

    def add_disliked_paper(self, user_id: str = DEFAULT_USER_ID, arxiv_id: str = None) -> bool:
        try:
            if arxiv_id is None:
                logger.error("arxiv_id is required")
                return False
                
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM user_liked_papers WHERE user_id = ? AND arxiv_id = ?
                ''', (user_id, arxiv_id))
                
                cursor.execute('''
                    INSERT OR IGNORE INTO user_disliked_papers (user_id, arxiv_id)
                    VALUES (?, ?)
                ''', (user_id, arxiv_id))
                
                conn.commit()
                logger.info(f"Paper {arxiv_id} added to disliked list for user: {user_id}")
                return True
        except Exception as e:
            logger.error(f"Error adding disliked paper: {str(e)}")
            return False

    def remove_disliked_paper(self, user_id: str = DEFAULT_USER_ID, arxiv_id: str = None) -> bool:
        try:
            if arxiv_id is None:
                logger.error("arxiv_id is required")
                return False
                
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM user_disliked_papers WHERE user_id = ? AND arxiv_id = ?
                ''', (user_id, arxiv_id))
                
                conn.commit()
                logger.info(f"Paper {arxiv_id} removed from disliked list for user: {user_id}")
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error removing disliked paper: {str(e)}")
            return False

    def get_disliked_papers(self, user_id: str = DEFAULT_USER_ID) -> List[str]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT arxiv_id FROM user_disliked_papers WHERE user_id = ? ORDER BY created_at DESC
                ''', (user_id,))
                
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error getting disliked papers: {str(e)}")
            return []

    def is_disliked_paper(self, user_id: str = DEFAULT_USER_ID, arxiv_id: str = None) -> bool:
        try:
            if arxiv_id is None:
                return False
                
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*) FROM user_disliked_papers WHERE user_id = ? AND arxiv_id = ?
                ''', (user_id, arxiv_id))
                
                return cursor.fetchone()[0] > 0
        except Exception as e:
            logger.error(f"Error checking disliked paper: {str(e)}")
            return False

    def get_user_preferences(self, user_id: str = DEFAULT_USER_ID) -> Dict[str, Any]:
        try:
            return {
                'user_id': user_id,
                'liked_papers': self.get_liked_papers(user_id),
                'disliked_papers': self.get_disliked_papers(user_id)
            }
        except Exception as e:
            logger.error(f"Error getting user preferences: {str(e)}")
            return {
                'user_id': user_id,
                'liked_papers': [],
                'disliked_papers': []
            }

    def get_latest_user_preference_timestamp(self, user_id: str = DEFAULT_USER_ID) -> Optional[str]:
        """
        返回该用户最新一次偏好的创建时间。
        用于判断兴趣向量是否已经过期。
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''
                    SELECT MAX(latest_at) FROM (
                        SELECT created_at AS latest_at FROM user_liked_papers WHERE user_id = ?
                        UNION ALL
                        SELECT created_at AS latest_at FROM user_disliked_papers WHERE user_id = ?
                    )
                    ''',
                    (user_id, user_id),
                )
                row = cursor.fetchone()
                return row[0] if row and row[0] else None
        except Exception as e:
            logger.error(f"Error getting latest preference timestamp: {str(e)}")
            return None

    def add_paper(self, paper: Dict[str, Any]) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO arxiv_papers 
                    (arxiv_id, title, authors, abstract, categories, published_date, url, embedding_id, embedding_model, embedded_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    paper.get('arxiv_id'),
                    paper.get('title'),
                    paper.get('authors'),
                    paper.get('abstract'),
                    paper.get('categories'),
                    paper.get('published_date'),
                    paper.get('url'),
                    paper.get('embedding_id'),
                    paper.get('embedding_model')
                ))
                
                conn.commit()
                logger.info(f"Paper added: {paper.get('arxiv_id')}")
                return True
        except Exception as e:
            logger.error(f"Error adding paper: {str(e)}")
            return False

    def get_paper(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT arxiv_id, title, authors, abstract, categories, published_date, url, embedding_id, embedding_model, created_at
                    FROM arxiv_papers WHERE arxiv_id = ?
                ''', (arxiv_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'arxiv_id': row[0],
                        'title': row[1],
                        'authors': row[2],
                        'abstract': row[3],
                        'categories': row[4],
                        'published_date': row[5],
                        'url': row[6],
                        'embedding_id': row[7],
                        'embedding_model': row[8],
                        'created_at': row[9]
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting paper: {str(e)}")
            return None

    def search_papers_by_category(self, category: str) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT arxiv_id, title, authors, abstract, categories, published_date, url, embedding_id, created_at
                    FROM arxiv_papers WHERE categories LIKE ?
                ''', (f'%{category}%',))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'arxiv_id': row[0],
                        'title': row[1],
                        'authors': row[2],
                        'abstract': row[3],
                        'categories': row[4],
                        'published_date': row[5],
                        'url': row[6],
                        'embedding_id': row[7],
                        'created_at': row[8]
                    })
                return results
        except Exception as e:
            logger.error(f"Error searching papers by category: {str(e)}")
            return []

    def get_all_papers(self) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT arxiv_id, title, authors, abstract, categories, published_date, url, embedding_id, created_at
                    FROM arxiv_papers ORDER BY published_date DESC
                ''')
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'arxiv_id': row[0],
                        'title': row[1],
                        'authors': row[2],
                        'abstract': row[3],
                        'categories': row[4],
                        'published_date': row[5],
                        'url': row[6],
                        'embedding_id': row[7],
                        'created_at': row[8]
                    })
                return results
        except Exception as e:
            logger.error(f"Error getting all papers: {str(e)}")
            return []

    def delete_paper(self, arxiv_id: str) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('DELETE FROM arxiv_papers WHERE arxiv_id = ?', (arxiv_id,))
                conn.commit()
                logger.info(f"Paper deleted: {arxiv_id}")
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error deleting paper: {str(e)}")
            return False

    def update_paper_embedding(self, arxiv_id: str, embedding_id: int, embedding_model: str = None) -> bool:
        """
        更新论文的embedding_id和embedding_model信息
        
        参数:
            arxiv_id: 论文的arXiv ID
            embedding_id: 向量数据库中的embedding ID
            embedding_model: 使用的嵌入模型名称
            
        返回:
            是否更新成功
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                if embedding_model:
                    cursor.execute('''
                        UPDATE arxiv_papers 
                        SET embedding_id = ?, embedding_model = ?, embedded_at = CURRENT_TIMESTAMP
                        WHERE arxiv_id = ?
                    ''', (str(embedding_id), embedding_model, arxiv_id))
                else:
                    cursor.execute('''
                        UPDATE arxiv_papers 
                        SET embedding_id = ?, embedded_at = CURRENT_TIMESTAMP
                        WHERE arxiv_id = ?
                    ''', (str(embedding_id), arxiv_id))
                
                conn.commit()
                logger.info(f"Paper {arxiv_id} embedding updated with id: {embedding_id}")
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating paper embedding: {str(e)}")
            return False

    def save_user_interest_vector(
        self,
        user_id: str,
        vector_data: List[float],
        paper_count: int,
        embedding_model: str,
        vector_dimension: int
    ) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO user_interest_vectors 
                    (user_id, vector_data, paper_count, embedding_model, vector_dimension, updated_at)
                    VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ''', (
                    user_id,
                    json.dumps(vector_data),
                    paper_count,
                    embedding_model,
                    vector_dimension
                ))
                
                conn.commit()
                logger.info(f"User interest vector saved for user: {user_id}")
                return True
        except Exception as e:
            logger.error(f"Error saving user interest vector: {str(e)}")
            return False

    def get_user_interest_vector(self, user_id: str = DEFAULT_USER_ID) -> Optional[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT user_id, vector_data, paper_count, embedding_model, vector_dimension, created_at, updated_at
                    FROM user_interest_vectors WHERE user_id = ?
                ''', (user_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'user_id': row[0],
                        'vector_data': json.loads(row[1]),
                        'paper_count': row[2],
                        'embedding_model': row[3],
                        'vector_dimension': row[4],
                        'created_at': row[5],
                        'updated_at': row[6]
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting user interest vector: {str(e)}")
            return None

    def get_unlabeled_papers(self, user_id: str = DEFAULT_USER_ID) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT p.arxiv_id, p.title, p.abstract, p.authors, p.categories, p.published_date, p.url
                    FROM arxiv_papers p
                    LEFT JOIN user_liked_papers ulp ON p.arxiv_id = ulp.arxiv_id AND ulp.user_id = ?
                    LEFT JOIN user_disliked_papers udp ON p.arxiv_id = udp.arxiv_id AND udp.user_id = ?
                    WHERE ulp.arxiv_id IS NULL AND udp.arxiv_id IS NULL AND p.embedding_id IS NOT NULL
                    ORDER BY p.published_date DESC
                ''', (user_id, user_id))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'arxiv_id': row[0],
                        'title': row[1],
                        'abstract': row[2],
                        'authors': row[3],
                        'categories': row[4],
                        'published_date': row[5],
                        'url': row[6]
                    })
                return results
        except Exception as e:
            logger.error(f"Error getting unlabeled papers: {str(e)}")
            return []

    def get_paper_qa_index(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT arxiv_id, collection_name, status, chunk_count, embedding_model, pdf_path, created_at, updated_at
                    FROM paper_qa_index WHERE arxiv_id = ?
                ''', (arxiv_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'arxiv_id': row[0],
                        'collection_name': row[1],
                        'status': row[2],
                        'chunk_count': row[3],
                        'embedding_model': row[4],
                        'pdf_path': row[5],
                        'created_at': row[6],
                        'updated_at': row[7]
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting paper QA index: {str(e)}")
            return None

    def update_paper_qa_index(self, arxiv_id: str, **kwargs) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                update_fields = []
                update_values = []
                
                if 'collection_name' in kwargs:
                    update_fields.append('collection_name = ?')
                    update_values.append(kwargs['collection_name'])
                if 'status' in kwargs:
                    update_fields.append('status = ?')
                    update_values.append(kwargs['status'])
                if 'chunk_count' in kwargs:
                    update_fields.append('chunk_count = ?')
                    update_values.append(kwargs['chunk_count'])
                if 'embedding_model' in kwargs:
                    update_fields.append('embedding_model = ?')
                    update_values.append(kwargs['embedding_model'])
                if 'pdf_path' in kwargs:
                    update_fields.append('pdf_path = ?')
                    update_values.append(kwargs['pdf_path'])
                
                update_fields.append('updated_at = CURRENT_TIMESTAMP')
                update_values.append(arxiv_id)
                
                if update_fields:
                    cursor.execute(f'''
                        UPDATE paper_qa_index 
                        SET {", ".join(update_fields)}
                        WHERE arxiv_id = ?
                    ''', update_values)
                
                conn.commit()
                logger.info(f"Paper QA index updated for: {arxiv_id}")
                return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating paper QA index: {str(e)}")
            return False

    def insert_paper_qa_index(self, arxiv_id: str, **kwargs) -> bool:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                fields = ['arxiv_id']
                values = [arxiv_id]
                
                if 'collection_name' in kwargs:
                    fields.append('collection_name')
                    values.append(kwargs['collection_name'])
                if 'status' in kwargs:
                    fields.append('status')
                    values.append(kwargs['status'])
                if 'chunk_count' in kwargs:
                    fields.append('chunk_count')
                    values.append(kwargs['chunk_count'])
                if 'embedding_model' in kwargs:
                    fields.append('embedding_model')
                    values.append(kwargs['embedding_model'])
                if 'pdf_path' in kwargs:
                    fields.append('pdf_path')
                    values.append(kwargs['pdf_path'])
                
                placeholders = ', '.join(['?' for _ in values])
                
                cursor.execute(f'''
                    INSERT OR REPLACE INTO paper_qa_index ({", ".join(fields)})
                    VALUES ({placeholders})
                ''', values)
                
                conn.commit()
                logger.info(f"Paper QA index inserted for: {arxiv_id}")
                return True
        except Exception as e:
            logger.error(f"Error inserting paper QA index: {str(e)}")
            return False
