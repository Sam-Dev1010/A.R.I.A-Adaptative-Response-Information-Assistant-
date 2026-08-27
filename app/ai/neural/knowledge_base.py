"""Base de conocimiento para A.R.I.A: almacena hechos con relaciones.

Permite buscar, inferir y aprender nuevo conocimiento de forma autónoma.
"""
import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class KnowledgeBase:
    """Base de conocimiento estructurada con SQLite."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    def open(self) -> "KnowledgeBase":
        """Abre la base de datos."""
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._create_tables()
        return self

    def _create_tables(self) -> None:
        """Crea las tablas necesarias."""
        conn = self._conn
        if conn is None:
            return

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'general',
                confidence REAL NOT NULL DEFAULT 1.0,
                source TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                type TEXT NOT NULL DEFAULT 'unknown',
                properties TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject_id INTEGER NOT NULL REFERENCES entities(id),
                predicate TEXT NOT NULL,
                object_id INTEGER NOT NULL REFERENCES entities(id),
                confidence REAL NOT NULL DEFAULT 1.0,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
            CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
            CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject_id);
            CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(object_id);
        """)
        conn.commit()

    def close(self) -> None:
        """Cierra la base de datos."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def _timestamp(self) -> str:
        return datetime.now(UTC).astimezone().isoformat(timespec="seconds")

    # --- Hechos ---

    def add_fact(
        self,
        content: str,
        category: str = "general",
        confidence: float = 1.0,
        source: str = "user",
    ) -> int:
        """Añade un hecho a la base de conocimiento."""
        conn = self._conn
        if conn is None:
            self.open()
            conn = self._conn
        if conn is None:
            return -1

        now = self._timestamp()
        with self._lock:
            cursor = conn.execute(
                "INSERT INTO facts (content, category, confidence, source, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (content, category, confidence, source, now, now),
            )
            conn.commit()
            return cursor.lastrowid or -1

    def search_facts(
        self,
        query: str,
        category: str | None = None,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Busca hechos por contenido (búsqueda de texto simple)."""
        conn = self._conn
        if conn is None:
            return []

        with self._lock:
            if category:
                rows = conn.execute(
                    "SELECT * FROM facts WHERE content LIKE ? AND category = ? "
                    "ORDER BY confidence DESC, created_at DESC LIMIT ?",
                    (f"%{query}%", category, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM facts WHERE content LIKE ? "
                    "ORDER BY confidence DESC, created_at DESC LIMIT ?",
                    (f"%{query}%", limit),
                ).fetchall()

        return [dict(row) for row in rows]

    def get_facts_by_category(self, category: str) -> list[dict[str, Any]]:
        """Obtiene todos los hechos de una categoría."""
        conn = self._conn
        if conn is None:
            return []

        with self._lock:
            rows = conn.execute(
                "SELECT * FROM facts WHERE category = ? ORDER BY created_at DESC",
                (category,),
            ).fetchall()

        return [dict(row) for row in rows]

    def update_fact_confidence(self, fact_id: int, confidence: float) -> bool:
        """Actualiza la confianza de un hecho."""
        conn = self._conn
        if conn is None:
            return False

        with self._lock:
            conn.execute(
                "UPDATE facts SET confidence = ?, updated_at = ? WHERE id = ?",
                (confidence, self._timestamp(), fact_id),
            )
            conn.commit()
            return True

    def delete_fact(self, fact_id: int) -> bool:
        """Elimina un hecho."""
        conn = self._conn
        if conn is None:
            return False

        with self._lock:
            conn.execute("DELETE FROM facts WHERE id = ?", (fact_id,))
            conn.commit()
            return True

    # --- Entidades ---

    def add_entity(
        self,
        name: str,
        entity_type: str = "unknown",
        properties: dict | None = None,
    ) -> int:
        """Añade una entidad (persona, lugar, cosa, concepto)."""
        conn = self._conn
        if conn is None:
            self.open()
            conn = self._conn
        if conn is None:
            return -1

        now = self._timestamp()
        props_json = json.dumps(properties or {}, ensure_ascii=False)

        with self._lock:
            try:
                cursor = conn.execute(
                    "INSERT INTO entities (name, type, properties, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (name.lower(), entity_type, props_json, now),
                )
                conn.commit()
                return cursor.lastrowid or -1
            except sqlite3.IntegrityError:
                # Ya existe, devolver su ID
                row = conn.execute(
                    "SELECT id FROM entities WHERE name = ?", (name.lower(),)
                ).fetchone()
                return row["id"] if row else -1

    def get_entity(self, name: str) -> dict[str, Any] | None:
        """Obtiene una entidad por nombre."""
        conn = self._conn
        if conn is None:
            return None

        with self._lock:
            row = conn.execute(
                "SELECT * FROM entities WHERE name = ?", (name.lower(),)
            ).fetchone()

        if row:
            result = dict(row)
            result["properties"] = json.loads(result["properties"])
            return result
        return None

    def search_entities(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Busca entidades por nombre."""
        conn = self._conn
        if conn is None:
            return []

        with self._lock:
            rows = conn.execute(
                "SELECT * FROM entities WHERE name LIKE ? LIMIT ?",
                (f"%{query.lower()}%", limit),
            ).fetchall()

        results = []
        for row in rows:
            result = dict(row)
            result["properties"] = json.loads(result["properties"])
            results.append(result)
        return results

    # --- Relaciones ---

    def add_relation(
        self,
        subject_name: str,
        predicate: str,
        object_name: str,
        confidence: float = 1.0,
    ) -> int | None:
        """Añade una relación entre dos entidades."""
        # Crear entidades si no existen
        subject_id = self.add_entity(subject_name)
        object_id = self.add_entity(object_name)

        if subject_id < 0 or object_id < 0:
            return None

        conn = self._conn
        if conn is None:
            return None

        now = self._timestamp()
        with self._lock:
            cursor = conn.execute(
                "INSERT INTO relations (subject_id, predicate, object_id, confidence, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (subject_id, predicate.lower(), object_id, confidence, now),
            )
            conn.commit()
            return cursor.lastrowid

    def get_relations(
        self,
        subject_name: str | None = None,
        predicate: str | None = None,
        object_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Obtiene relaciones que coincidan con los criterios."""
        conn = self._conn
        if conn is None:
            return []

        query = """
            SELECT r.*, s.name as subject_name, o.name as object_name
            FROM relations r
            JOIN entities s ON r.subject_id = s.id
            JOIN entities o ON r.object_id = o.id
            WHERE 1=1
        """
        params: list[Any] = []

        if subject_name:
            query += " AND s.name = ?"
            params.append(subject_name.lower())
        if predicate:
            query += " AND r.predicate = ?"
            params.append(predicate.lower())
        if object_name:
            query += " AND o.name = ?"
            params.append(object_name.lower())

        with self._lock:
            rows = conn.execute(query, params).fetchall()

        return [dict(row) for row in rows]

    def infer(self, subject_name: str) -> list[str]:
        """Inferencia simple: devuelve todo lo que se sabe sobre una entidad."""
        facts = []

        # Buscar hechos que mencionen la entidad
        entity_facts = self.search_facts(subject_name)
        for fact in entity_facts:
            facts.append(fact["content"])

        # Buscar relaciones donde es sujeto
        as_subject = self.get_relations(subject_name=subject_name)
        for rel in as_subject:
            facts.append(f"{subject_name} {rel['predicate']} {rel['object_name']}")

        # Buscar relaciones donde es objeto
        as_object = self.get_relations(object_name=subject_name)
        for rel in as_object:
            facts.append(f"{rel['subject_name']} {rel['predicate']} {subject_name}")

        return facts

    def stats(self) -> dict[str, int]:
        """Estadísticas de la base de conocimiento."""
        conn = self._conn
        if conn is None:
            return {"facts": 0, "entities": 0, "relations": 0}

        with self._lock:
            facts = conn.execute("SELECT COUNT(*) as n FROM facts").fetchone()["n"]
            entities = conn.execute("SELECT COUNT(*) as n FROM entities").fetchone()["n"]
            relations = conn.execute("SELECT COUNT(*) as n FROM relations").fetchone()["n"]

        return {"facts": facts, "entities": entities, "relations": relations}
