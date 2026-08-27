"""Memoria persistente de SIA en SQLite (FASE 4).

Almacena tres cosas:
- Sesiones de conversación (una por ejecución del CLI).
- Mensajes user/assistant por sesión.
- Hechos (datos que SIA debe recordar a largo plazo).

El acceso es seguro desde hilos y la base de datos se crea/abre perezosamente
en el primer uso.
"""
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from app.ai.schemas import ChatMessage, ChatRole

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    origin TEXT NOT NULL DEFAULT '',
    shared_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""

# Orígenes de un hecho: el usuario lo dijo, SIA lo dedujo de la charla o lo
# aprendió investigando en internet.
_ORIGEN_USUARIO = ""
_ORIGEN_AUTO = "auto"
_ORIGEN_WEB = "web"
_ORIGENES_AUTONOMOS = (_ORIGEN_AUTO, _ORIGEN_WEB)


class MemoryManager:
    """Acceso a la memoria persistente de SIA (SQLite)."""

    def __init__(
        self,
        db_path: Path | str,
        *,
        max_sessions: int = 20,
        max_messages_per_session: int = 200,
        max_facts: int = 100,
    ) -> None:
        self._db_path = Path(db_path)
        self._max_sessions = max_sessions
        self._max_messages_per_session = max_messages_per_session
        self._max_facts = max_facts
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None

    # --- conexión ---------------------------------------------------------

    def open(self) -> "MemoryManager":
        """Abre (o crea) la base de datos. Idempotente."""
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                self._db_path, check_same_thread=False, timeout=10
            )
            self._conn.row_factory = sqlite3.Row
            with self._lock:
                # WAL: permite varios procesos (servicio de voz, web, CLI)
                # escribiendo a la vez sin "database is locked".
                self._conn.execute("PRAGMA journal_mode = WAL")
                self._conn.execute("PRAGMA foreign_keys = ON")
                self._conn.executescript(_SCHEMA)
                self._migrar(self._conn)
                self._conn.commit()
        return self

    @staticmethod
    def _migrar(conn: sqlite3.Connection) -> None:
        """Añade columnas nuevas a bases creadas con versiones anteriores."""
        columnas = {row["name"] for row in conn.execute("PRAGMA table_info(facts)")}
        if "origin" not in columnas:
            conn.execute(
                "ALTER TABLE facts ADD COLUMN origin TEXT NOT NULL DEFAULT ''"
            )
        if "shared_at" not in columnas:
            conn.execute("ALTER TABLE facts ADD COLUMN shared_at TEXT")

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(self, *exc_info) -> None:
        self.close()

    def _conn_or_open(self) -> sqlite3.Connection:
        return self.open()._conn

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(UTC).astimezone().isoformat(timespec="seconds")

    # --- sesiones ---------------------------------------------------------

    def start_session(self, title: str = "") -> int:
        """Inicia una sesión nueva (las anteriores se conservan como historial)."""
        conn = self._conn_or_open()
        with self._lock:
            cursor = conn.execute(
                "INSERT INTO sessions (title, created_at) VALUES (?, ?)",
                (title, self._timestamp()),
            )
            session_id = cursor.lastrowid
            self._prune_sessions(conn)
            conn.commit()
        return session_id

    def _prune_sessions(self, conn: sqlite3.Connection) -> None:
        """Borra las sesiones más antiguas que superen el máximo."""
        conn.execute(
            """
            DELETE FROM sessions
            WHERE id NOT IN (
                SELECT id FROM sessions ORDER BY id DESC LIMIT ?
            )
            """,
            (self._max_sessions,),
        )

    # --- mensajes ---------------------------------------------------------

    def add_message(self, role: ChatRole | str, content: str) -> None:
        """Añade un mensaje a la sesión actual (crea una si no existe)."""
        role = ChatRole(role).value
        conn = self._conn_or_open()
        with self._lock:
            row = conn.execute(
                "SELECT id FROM sessions ORDER BY id DESC LIMIT 1"
            ).fetchone()
            session_id = row["id"] if row else self._insert_session(conn)

            conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, role, content, self._timestamp()),
            )
            self._prune_messages(conn, session_id)
            conn.commit()

    def add_exchange(self, user_content: str, assistant_content: str) -> None:
        """Guarda un intercambio completo user/assistant."""
        self.add_message(ChatRole.USER, user_content)
        self.add_message(ChatRole.ASSISTANT, assistant_content)

    def _insert_session(self, conn: sqlite3.Connection) -> int:
        cursor = conn.execute(
            "INSERT INTO sessions (title, created_at) VALUES ('', ?)",
            (self._timestamp(),),
        )
        return cursor.lastrowid

    def _prune_messages(self, conn: sqlite3.Connection, session_id: int) -> None:
        conn.execute(
            """
            DELETE FROM messages
            WHERE session_id = ?
              AND id NOT IN (
                  SELECT id FROM messages
                  WHERE session_id = ?
                  ORDER BY id DESC LIMIT ?
              )
            """,
            (session_id, session_id, self._max_messages_per_session),
        )

    def recent_messages(self, limit: int = 20) -> list[ChatMessage]:
        """Últimos mensajes de la sesión más reciente con historial.

        Se usa para dar contexto al LLM al arrancar una nueva conversación.
        """
        conn = self._conn_or_open()
        limit -= limit % 2  # siempre pares user/assistant
        if limit <= 0:
            return []
        with self._lock:
            row = conn.execute(
                "SELECT session_id FROM messages ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return []
            rows = conn.execute(
                "SELECT role, content FROM messages "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (row["session_id"], limit),
            ).fetchall()
        return [
            ChatMessage(role=ChatRole(r["role"]), content=r["content"])
            for r in reversed(rows)
        ]

    def count_messages(self) -> int:
        conn = self._conn_or_open()
        with self._lock:
            row = conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()
        return row["n"]

    # --- hechos (memoria a largo plazo) -----------------------------------

    def remember(self, content: str, *, origin: str = _ORIGEN_USUARIO) -> None:
        """Guarda un hecho. Si ya existía, lo ignora (idempotente).

        ``origin``: '' = lo dijo el usuario, 'auto' = deducido de la charla,
        'web' = aprendido investigando en internet.
        """
        content = content.strip()
        if not content:
            return
        conn = self._conn_or_open()
        with self._lock:
            conn.execute(
                "INSERT OR IGNORE INTO facts (content, created_at, origin) "
                "VALUES (?, ?, ?)",
                (content, self._timestamp(), origin),
            )
            conn.execute(
                """
                DELETE FROM facts
                WHERE id NOT IN (
                    SELECT id FROM facts ORDER BY id DESC LIMIT ?
                )
                """,
                (self._max_facts,),
            )
            conn.commit()

    def facts(self) -> list[str]:
        conn = self._conn_or_open()
        with self._lock:
            rows = conn.execute(
                "SELECT content FROM facts ORDER BY id"
            ).fetchall()
        return [row["content"] for row in rows]

    def forget(self, content: str) -> bool:
        """Borra un hecho. Devuelve True si existía."""
        conn = self._conn_or_open()
        with self._lock:
            cursor = conn.execute("DELETE FROM facts WHERE content = ?", (content,))
            conn.commit()
        return cursor.rowcount > 0

    # --- descubrimientos autónomos pendientes de contar --------------------

    def pending_discoveries(self, limit: int = 3) -> list[str]:
        """Hechos aprendidos por sí misma que aún no ha compartido con nadie."""
        conn = self._conn_or_open()
        placeholders = ", ".join("?" for _ in _ORIGENES_AUTONOMOS)
        with self._lock:
            rows = conn.execute(
                f"""
                SELECT content FROM facts
                WHERE origin IN ({placeholders}) AND shared_at IS NULL
                ORDER BY id DESC LIMIT ?
                """,
                (*_ORIGENES_AUTONOMOS, limit),
            ).fetchall()
        return [row["content"] for row in rows]

    def mark_shared(self, contents: list[str]) -> None:
        """Marca hechos como ya contados (dejan de ser 'novedades')."""
        if not contents:
            return
        conn = self._conn_or_open()
        ahora = self._timestamp()
        with self._lock:
            conn.executemany(
                "UPDATE facts SET shared_at = ? WHERE content = ?",
                [(ahora, content) for content in contents],
            )
            conn.commit()

    # --- utilidades -------------------------------------------------------

    def stats(self) -> dict:
        conn = self._conn_or_open()
        with self._lock:
            sessions = conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"]
            messages = conn.execute("SELECT COUNT(*) AS n FROM messages").fetchone()["n"]
            facts = conn.execute("SELECT COUNT(*) AS n FROM facts").fetchone()["n"]
        return {"sessions": sessions, "messages": messages, "facts": facts}

    def clear(self) -> None:
        """Borra todo el contenido de la memoria (no la estructura)."""
        conn = self._conn_or_open()
        with self._lock:
            conn.execute("DELETE FROM facts")
            conn.execute("DELETE FROM messages")
            conn.execute("DELETE FROM sessions")
            conn.commit()