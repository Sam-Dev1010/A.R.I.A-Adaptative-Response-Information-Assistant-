"""Memoria semántica para A.R.I.A: busca por significado, no por palabras exactas.

Implementa embeddings simples para encontrar información relevante
incluso si las palabras no coinciden exactamente.
"""
import json
import math
import sqlite3
import threading
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class SemanticMemory:
    """Memoria con búsqueda semántica usando TF-IDF simplificado."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

        # Índice invertido para búsqueda semántica
        self._idf: dict[str, float] = {}
        self._doc_count = 0
        self._token_docs: dict[str, int] = defaultdict(int)

    def open(self) -> "SemanticMemory":
        """Abre la base de datos."""
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._create_tables()
            self._build_index()
        return self._conn

    def _create_tables(self) -> None:
        """Crea las tablas necesarias."""
        conn = self._conn
        if conn is None:
            return

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'general',
                importance REAL NOT NULL DEFAULT 0.5,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed TEXT,
                created_at TEXT NOT NULL,
                embedding TEXT
            );

            CREATE TABLE IF NOT EXISTS memory_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                memory_a_id INTEGER NOT NULL REFERENCES memories(id),
                memory_b_id INTEGER NOT NULL REFERENCES memories(id),
                strength REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
            CREATE INDEX IF NOT EXISTS idx_memory_links_a ON memory_links(memory_a_id);
            CREATE INDEX IF NOT EXISTS idx_memory_links_b ON memory_links(memory_b_id);
        """)
        conn.commit()

    def _build_index(self) -> None:
        """Construye el índice TF-IDF."""
        conn = self._conn
        if conn is None:
            return

        with self._lock:
            rows = conn.execute("SELECT id, content FROM memories").fetchall()
            self._doc_count = len(rows)

            for row in rows:
                tokens = self._tokenize(row["content"])
                unique_tokens = set(tokens)
                for token in unique_tokens:
                    self._token_docs[token] += 1

            # Calcular IDF
            for token, doc_freq in self._token_docs.items():
                self._idf[token] = math.log(self._doc_count / (1 + doc_freq)) + 1.0

            # Recompute embeddings for all stored memories with correct IDF
            all_rows = conn.execute("SELECT id, content FROM memories").fetchall()
            for row in all_rows:
                tfidf = self._compute_tfidf(row["content"])
                conn.execute(
                    "UPDATE memories SET embedding = ? WHERE id = ?",
                    (str(dict(tfidf)), row["id"]),
                )
            conn.commit()

    def close(self) -> None:
        """Cierra la base de datos."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _timestamp(self) -> str:
        return datetime.now(UTC).astimezone().isoformat(timespec="seconds")

    def _tokenize(self, text: str) -> list[str]:
        """Tokeniza texto para búsqueda."""
        text = text.lower()
        # Separar puntuación
        import re
        text = re.sub(r'([.,!?;:¿¡()"\'])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.split()

    def _compute_tfidf(self, text: str) -> dict[str, float]:
        """Computa el vector TF-IDF de un texto."""
        tokens = self._tokenize(text)
        if not tokens:
            return {}

        # TF: frecuencia de cada token
        tf: dict[str, int] = defaultdict(int)
        for token in tokens:
            tf[token] += 1

        # TF-IDF
        tfidf: dict[str, float] = {}
        for token, freq in tf.items():
            idf = self._idf.get(token, math.log(2))  # Default si es nuevo
            tfidf[token] = freq * idf

        return tfidf

    def _cosine_similarity(self, vec1: dict[str, float], vec2: dict[str, float]) -> float:
        """Calcula similitud coseno entre dos vectores TF-IDF."""
        if not vec1 or not vec2:
            return 0.0

        # Product punto
        intersection = set(vec1.keys()) & set(vec2.keys())
        dot_product = sum(vec1[k] * vec2[k] for k in intersection)

        # Magnitudes
        mag1 = math.sqrt(sum(v ** 2 for v in vec1.values()))
        mag2 = math.sqrt(sum(v ** 2 for v in vec2.values()))

        if mag1 == 0 or mag2 == 0:
            return 0.0

        return dot_product / (mag1 * mag2)

    def store(
        self,
        content: str,
        category: str = "general",
        importance: float = 0.5,
    ) -> int:
        """Almacena un recuerdo con embedding."""
        conn = self._conn
        if conn is None:
            self.open()
            conn = self._conn
        if conn is None:
            return -1

        # Calcular embedding
        tfidf = self._compute_tfidf(content)
        embedding_str = str(dict(tfidf))

        now = self._timestamp()
        with self._lock:
            cursor = conn.execute(
                "INSERT INTO memories (content, category, importance, created_at, embedding) "
                "VALUES (?, ?, ?, ?, ?)",
                (content, category, importance, now, embedding_str),
            )
            conn.commit()
            memory_id = cursor.lastrowid

            # Actualizar índice
            tokens = set(self._tokenize(content))
            for token in tokens:
                self._token_docs[token] += 1
            self._doc_count += 1
            for token in tokens:
                if token not in self._idf:
                    self._idf[token] = math.log(self._doc_count / (1 + self._token_docs[token])) + 1.0

            return memory_id or -1

    def search(
        self,
        query: str,
        limit: int = 5,
        category: str | None = None,
        min_importance: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Búsqueda semántica por significado."""
        conn = self._conn
        if conn is None:
            return []

        query_tfidf = self._compute_tfidf(query)

        # Obtener candidatos
        with self._lock:
            if category:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE category = ? AND importance >= ?",
                    (category, min_importance),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE importance >= ?",
                    (min_importance,),
                ).fetchall()

        # Calcular similitud y ordenar
        results = []
        for row in rows:
            embedding_str = row["embedding"]
            if embedding_str:
                try:
                    doc_tfidf = json.loads(embedding_str.replace("'", '"'))
                    similarity = self._cosine_similarity(query_tfidf, doc_tfidf)
                except Exception:
                    similarity = 0.0
            else:
                # Fallback: coincidencia de palabras
                tokens = set(self._tokenize(row["content"]))
                query_tokens = set(self._tokenize(query))
                if tokens and query_tokens:
                    similarity = len(tokens & query_tokens) / len(tokens | query_tokens)
                else:
                    similarity = 0.0

            if similarity > 0.1:  # Umbral mínimo
                results.append({
                    "id": row["id"],
                    "content": row["content"],
                    "category": row["category"],
                    "importance": row["importance"],
                    "similarity": similarity,
                })

        # Ordenar por similitud
        results.sort(key=lambda x: x["similarity"], reverse=True)

        # Actualizar contadores de acceso
        with self._lock:
            for result in results[:limit]:
                conn.execute(
                    "UPDATE memories SET access_count = access_count + 1, "
                    "last_accessed = ? WHERE id = ?",
                    (self._timestamp(), result["id"]),
                )
            conn.commit()

        return results[:limit]

    def link(self, memory_a_id: int, memory_b_id: int, strength: float = 0.5) -> bool:
        """Crea un enlace entre dos recuerdos."""
        conn = self._conn
        if conn is None:
            return False

        now = self._timestamp()
        with self._lock:
            try:
                conn.execute(
                    "INSERT INTO memory_links (memory_a_id, memory_b_id, strength, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (memory_a_id, memory_b_id, strength, now),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def get_related(self, memory_id: int, limit: int = 5) -> list[dict[str, Any]]:
        """Obtiene recuerdos relacionados."""
        conn = self._conn
        if conn is None:
            return []

        with self._lock:
            rows = conn.execute(
                """
                SELECT m.*, ml.strength
                FROM memories m
                JOIN memory_links ml ON (ml.memory_a_id = ? OR ml.memory_b_id = ?)
                WHERE m.id != ?
                ORDER BY ml.strength DESC
                LIMIT ?
                """,
                (memory_id, memory_id, memory_id, limit),
            ).fetchall()

        return [dict(row) for row in rows]

    def get_by_category(self, category: str, limit: int = 20) -> list[dict[str, Any]]:
        """Obtiene todos los recuerdos de una categoría."""
        conn = self._conn
        if conn is None:
            return []

        with self._lock:
            rows = conn.execute(
                "SELECT * FROM memories WHERE category = ? "
                "ORDER BY importance DESC, access_count DESC LIMIT ?",
                (category, limit),
            ).fetchall()

        return [dict(row) for row in rows]

    def forget(self, memory_id: int) -> bool:
        """Elimina un recuerdo."""
        conn = self._conn
        if conn is None:
            return False

        with self._lock:
            conn.execute("DELETE FROM memory_links WHERE memory_a_id = ? OR memory_b_id = ?",
                         (memory_id, memory_id))
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
            return True

    def stats(self) -> dict[str, int]:
        """Estadísticas de la memoria."""
        conn = self._conn
        if conn is None:
            return {"memories": 0, "links": 0, "categories": 0}

        with self._lock:
            memories = conn.execute("SELECT COUNT(*) as n FROM memories").fetchone()["n"]
            links = conn.execute("SELECT COUNT(*) as n FROM memory_links").fetchone()["n"]
            categories = conn.execute(
                "SELECT COUNT(DISTINCT category) as n FROM memories"
            ).fetchone()["n"]

        return {"memories": memories, "links": links, "categories": categories}

    def consolidate(self) -> int:
        """Consolida recuerdos: fusiona duplicados y fortalece enlaces."""
        conn = self._conn
        if conn is None:
            return 0

        consolidated = 0

        # Buscar recuerdos similares y crear enlaces
        with self._lock:
            rows = conn.execute(
                "SELECT id, content, embedding FROM memories"
            ).fetchall()

        for i, row_a in enumerate(rows):
            if not row_a["embedding"]:
                continue

            try:
                tfidf_a = json.loads(row_a["embedding"].replace("'", '"'))
            except Exception:
                continue

            for row_b in rows[i + 1:]:
                if not row_b["embedding"]:
                    continue

                try:
                    tfidf_b = json.loads(row_b["embedding"].replace("'", '"'))
                except Exception:
                    continue

                similarity = self._cosine_similarity(tfidf_a, tfidf_b)
                if similarity > 0.5:  # Umbral para considerar relacionados
                    self.link(row_a["id"], row_b["id"], strength=similarity)
                    consolidated += 1

        return consolidated
