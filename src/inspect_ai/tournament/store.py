import json
import sqlite3
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from sqlite3 import Connection, Row
from typing import Any, Iterator, Literal, Mapping

from .config import TournamentConfig, load_tournament_config
from .types import ModelRating, default_project_id, model_id

SCHEMA_VERSION = 1
TableName = Literal[
    "models",
    "prompts",
    "responses",
    "matches",
    "judgments",
    "ratings",
    "ratings_history",
    "run_state",
]

MIGRATIONS: dict[int, str] = {
    1: """
    CREATE TABLE IF NOT EXISTS models (
      model_id TEXT PRIMARY KEY,
      model_name TEXT NOT NULL UNIQUE,
      added_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      active INTEGER NOT NULL DEFAULT 1,
      tags_json TEXT
    );

    CREATE TABLE IF NOT EXISTS prompts (
      prompt_id TEXT PRIMARY KEY,
      prompt_text TEXT NOT NULL,
      metadata_json TEXT
    );

    CREATE TABLE IF NOT EXISTS responses (
      response_id TEXT PRIMARY KEY,
      model_id TEXT NOT NULL,
      prompt_id TEXT NOT NULL,
      response_text TEXT NOT NULL,
      source_log TEXT,
      sample_id TEXT,
      sample_uuid TEXT,
      created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      UNIQUE(model_id, prompt_id),
      FOREIGN KEY(model_id) REFERENCES models(model_id),
      FOREIGN KEY(prompt_id) REFERENCES prompts(prompt_id)
    );

    CREATE TABLE IF NOT EXISTS matches (
      match_id TEXT PRIMARY KEY,
      model_a TEXT NOT NULL,
      model_b TEXT NOT NULL,
      prompt_id TEXT NOT NULL,
      response_a_id TEXT NOT NULL,
      response_b_id TEXT NOT NULL,
      batch_id TEXT NOT NULL,
      round_index INTEGER NOT NULL,
      status TEXT NOT NULL DEFAULT 'scheduled',
      scheduled_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY(model_a) REFERENCES models(model_id),
      FOREIGN KEY(model_b) REFERENCES models(model_id),
      FOREIGN KEY(prompt_id) REFERENCES prompts(prompt_id),
      FOREIGN KEY(response_a_id) REFERENCES responses(response_id),
      FOREIGN KEY(response_b_id) REFERENCES responses(response_id)
    );

    CREATE TABLE IF NOT EXISTS judgments (
      judgment_id TEXT PRIMARY KEY,
      match_id TEXT NOT NULL,
      side TEXT NOT NULL,
      decision TEXT NOT NULL,
      judge_model TEXT NOT NULL,
      explanation TEXT,
      raw_completion TEXT,
      source_log TEXT,
      sample_uuid TEXT,
      judged_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY(match_id) REFERENCES matches(match_id)
    );

    CREATE TABLE IF NOT EXISTS ratings (
      model_id TEXT PRIMARY KEY,
      mu REAL NOT NULL,
      sigma REAL NOT NULL,
      games INTEGER NOT NULL DEFAULT 0,
      wins INTEGER NOT NULL DEFAULT 0,
      losses INTEGER NOT NULL DEFAULT 0,
      ties INTEGER NOT NULL DEFAULT 0,
      updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY(model_id) REFERENCES models(model_id)
    );

    CREATE TABLE IF NOT EXISTS ratings_history (
      step_id INTEGER NOT NULL,
      model_id TEXT NOT NULL,
      mu REAL NOT NULL,
      sigma REAL NOT NULL,
      games INTEGER NOT NULL,
      timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY(step_id, model_id),
      FOREIGN KEY(model_id) REFERENCES models(model_id)
    );

    CREATE TABLE IF NOT EXISTS run_state (
      key TEXT PRIMARY KEY,
      value TEXT NOT NULL,
      updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_responses_prompt ON responses(prompt_id);
    CREATE INDEX IF NOT EXISTS idx_matches_pair ON matches(model_a, model_b);
    CREATE INDEX IF NOT EXISTS idx_matches_batch ON matches(batch_id);
    CREATE INDEX IF NOT EXISTS idx_judgments_match ON judgments(match_id);
    CREATE INDEX IF NOT EXISTS idx_ratings_sigma ON ratings(sigma);
    """,
}


class TournamentStore(AbstractContextManager["TournamentStore"]):
    """Persistent tournament state store."""

    def __init__(self, path: str | Path):
        db_path = Path(path)
        self.db_path = db_path if db_path.suffix == ".db" else db_path / "tournament.db"
        self.state_dir = self.db_path.parent
        self._conn: Connection | None = None

    def __enter__(self) -> "TournamentStore":
        self.open()
        return self

    def __exit__(self, *excinfo: Any) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        conn = self.connection()
        cursor = conn.execute("PRAGMA user_version")
        return int(cursor.fetchone()[0])

    def open(self) -> None:
        if self._conn is not None:
            return

        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path.as_posix())
        self._conn.row_factory = Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._apply_migrations()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def connection(self) -> Connection:
        if self._conn is None:
            raise RuntimeError("TournamentStore is not open. Use it as a context manager.")
        return self._conn

    def initialize_from_config(self, config: TournamentConfig) -> None:
        conn = self.connection()
        prompt_ids = [prompt.id for prompt in config.prompts]

        with self.transaction():
            for name in config.contestant_models:
                model_identifier = model_id(name)
                conn.execute(
                    """
                    INSERT INTO models (model_id, model_name, active)
                    VALUES (?, ?, 1)
                    ON CONFLICT(model_id) DO UPDATE SET
                      model_name = excluded.model_name,
                      active = 1
                    """,
                    (model_identifier, name),
                )
                conn.execute(
                    """
                    INSERT INTO ratings (model_id, mu, sigma, games, wins, losses, ties)
                    VALUES (?, ?, ?, 0, 0, 0, 0)
                    ON CONFLICT(model_id) DO NOTHING
                    """,
                    (model_identifier, config.rating_params.mu, config.rating_params.sigma),
                )

            for prompt in config.prompts:
                metadata_json = (
                    json.dumps(prompt.metadata, sort_keys=True)
                    if prompt.metadata is not None
                    else None
                )
                conn.execute(
                    """
                    INSERT INTO prompts (prompt_id, prompt_text, metadata_json)
                    VALUES (?, ?, ?)
                    ON CONFLICT(prompt_id) DO UPDATE SET
                      prompt_text = excluded.prompt_text,
                      metadata_json = excluded.metadata_json
                    """,
                    (prompt.id, prompt.text, metadata_json),
                )

            existing_project_id = self.get_run_state("project_id")
            project_id = (
                existing_project_id
                if existing_project_id is not None
                else (
                    config.project_id
                    or default_project_id(
                        config.contestant_models, prompt_ids, seed=config.seed
                    )
                )
            )
            self.set_run_state("project_id", project_id, commit=False)
            self.set_run_state("seed", str(config.seed), commit=False)
            self.set_run_state("schema_version", str(self.schema_version), commit=False)

    def get_run_state(self, key: str) -> str | None:
        conn = self.connection()
        cursor = conn.execute("SELECT value FROM run_state WHERE key = ?", (key,))
        row = cursor.fetchone()
        return str(row["value"]) if row is not None else None

    def set_run_state(self, key: str, value: str, *, commit: bool = True) -> None:
        conn = self.connection()
        conn.execute(
            """
            INSERT INTO run_state (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
              value = excluded.value,
              updated_at = CURRENT_TIMESTAMP
            """,
            (key, value),
        )
        if commit:
            conn.commit()

    def table_count(self, table_name: TableName) -> int:
        conn = self.connection()
        cursor = conn.execute(f"SELECT COUNT(*) AS count FROM {table_name}")
        return int(cursor.fetchone()["count"])

    def model_identifier(self, model_name: str) -> str | None:
        conn = self.connection()
        cursor = conn.execute(
            "SELECT model_id FROM models WHERE model_name = ?", (model_name,)
        )
        row = cursor.fetchone()
        return str(row["model_id"]) if row is not None else None

    def model_names_by_id(self) -> dict[str, str]:
        """Load model names keyed by model_id."""
        rows = self.connection().execute(
            "SELECT model_id, model_name FROM models ORDER BY model_name"
        ).fetchall()
        return {str(row["model_id"]): str(row["model_name"]) for row in rows}

    def upsert_response(
        self,
        *,
        response_id: str,
        model_id: str,
        prompt_id: str,
        response_text: str,
        source_log: str | None,
        sample_id: str | None,
        sample_uuid: str | None,
        commit: bool = True,
    ) -> bool:
        """Insert or update a response and return whether it was newly inserted."""
        conn = self.connection()
        existing = conn.execute(
            "SELECT response_id FROM responses WHERE model_id = ? AND prompt_id = ?",
            (model_id, prompt_id),
        ).fetchone()
        inserted = existing is None
        conn.execute(
            """
            INSERT INTO responses (
              response_id, model_id, prompt_id, response_text, source_log, sample_id, sample_uuid
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(model_id, prompt_id) DO UPDATE SET
              response_id = excluded.response_id,
              response_text = excluded.response_text,
              source_log = excluded.source_log,
              sample_id = excluded.sample_id,
              sample_uuid = excluded.sample_uuid
            """,
            (
                response_id,
                model_id,
                prompt_id,
                response_text,
                source_log,
                sample_id,
                sample_uuid,
            ),
        )
        if commit:
            conn.commit()
        return inserted

    def missing_prompt_ids_by_model(
        self, model_names: list[str], prompt_ids: list[str]
    ) -> dict[str, list[str]]:
        """Return missing prompt ids for each model name."""
        conn = self.connection()
        rows = conn.execute(
            """
            SELECT m.model_name AS model_name, r.prompt_id AS prompt_id
            FROM responses r
            JOIN models m ON m.model_id = r.model_id
            """
        ).fetchall()
        present = {(str(row["model_name"]), str(row["prompt_id"])) for row in rows}
        return {
            model_name: [
                prompt_id
                for prompt_id in prompt_ids
                if (model_name, prompt_id) not in present
            ]
            for model_name in model_names
        }

    def upsert_match(
        self,
        *,
        match_id: str,
        model_a: str,
        model_b: str,
        prompt_id: str,
        response_a_id: str,
        response_b_id: str,
        batch_id: str,
        round_index: int,
        status: str = "scheduled",
        commit: bool = True,
    ) -> None:
        """Insert or update a match row."""
        conn = self.connection()
        conn.execute(
            """
            INSERT INTO matches (
              match_id, model_a, model_b, prompt_id, response_a_id, response_b_id,
              batch_id, round_index, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(match_id) DO UPDATE SET
              status = excluded.status
            """,
            (
                match_id,
                model_a,
                model_b,
                prompt_id,
                response_a_id,
                response_b_id,
                batch_id,
                round_index,
                status,
            ),
        )
        if commit:
            conn.commit()

    def set_match_status(
        self,
        match_id: str,
        status: str,
        *,
        commit: bool = True,
    ) -> None:
        """Update status for one match row."""
        conn = self.connection()
        conn.execute(
            "UPDATE matches SET status = ? WHERE match_id = ?",
            (status, match_id),
        )
        if commit:
            conn.commit()

    def set_batch_match_status(
        self,
        *,
        batch_id: str,
        status: str,
        from_statuses: list[str] | None = None,
        commit: bool = True,
    ) -> None:
        """Update status for all matches in a batch."""
        conn = self.connection()
        if from_statuses is None or len(from_statuses) == 0:
            conn.execute(
                "UPDATE matches SET status = ? WHERE batch_id = ?",
                (status, batch_id),
            )
        else:
            placeholders = ", ".join(["?"] * len(from_statuses))
            conn.execute(
                f"""
                UPDATE matches
                SET status = ?
                WHERE batch_id = ?
                  AND status IN ({placeholders})
                """,
                (status, batch_id, *from_statuses),
            )
        if commit:
            conn.commit()

    def load_batch_matches(
        self,
        batch_id: str,
        *,
        statuses: list[str] | None = None,
    ) -> list[Row]:
        """Load matches with prompt/response payload needed for judging."""
        conn = self.connection()
        status_clause = ""
        params: list[str] = [batch_id]
        if statuses is not None and len(statuses) > 0:
            placeholders = ", ".join(["?"] * len(statuses))
            status_clause = f"AND m.status IN ({placeholders})"
            params.extend(statuses)

        return list(
            conn.execute(
                f"""
                SELECT
                  m.match_id,
                  m.batch_id,
                  m.round_index,
                  m.model_a AS model_a_id,
                  m.model_b AS model_b_id,
                  ma.model_name AS model_a_name,
                  mb.model_name AS model_b_name,
                  m.prompt_id,
                  m.status,
                  p.prompt_text,
                  ra.response_text AS response_a_text,
                  rb.response_text AS response_b_text
                FROM matches m
                JOIN models ma ON ma.model_id = m.model_a
                JOIN models mb ON mb.model_id = m.model_b
                JOIN prompts p ON p.prompt_id = m.prompt_id
                JOIN responses ra ON ra.response_id = m.response_a_id
                JOIN responses rb ON rb.response_id = m.response_b_id
                WHERE m.batch_id = ?
                {status_clause}
                ORDER BY m.match_id
                """,
                params,
            ).fetchall()
        )

    def load_batch_judgments(self, batch_id: str) -> list[Row]:
        """Load judgments for all matches in a batch."""
        conn = self.connection()
        return list(
            conn.execute(
                """
                SELECT
                  m.match_id,
                  m.model_a,
                  m.model_b,
                  j.side,
                  j.decision
                FROM matches m
                LEFT JOIN judgments j ON j.match_id = m.match_id
                WHERE m.batch_id = ?
                ORDER BY m.match_id, j.side
                """,
                (batch_id,),
            ).fetchall()
        )

    def upsert_judgment(
        self,
        *,
        judgment_id: str,
        match_id: str,
        side: str,
        decision: str,
        judge_model: str,
        explanation: str | None,
        raw_completion: str | None,
        source_log: str | None,
        sample_uuid: str | None,
        commit: bool = True,
    ) -> None:
        """Insert or update one judgment row."""
        conn = self.connection()
        conn.execute(
            """
            INSERT INTO judgments (
              judgment_id, match_id, side, decision, judge_model, explanation,
              raw_completion, source_log, sample_uuid
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(judgment_id) DO UPDATE SET
              decision = excluded.decision,
              judge_model = excluded.judge_model,
              explanation = excluded.explanation,
              raw_completion = excluded.raw_completion,
              source_log = excluded.source_log,
              sample_uuid = excluded.sample_uuid,
              judged_at = CURRENT_TIMESTAMP
            """,
            (
                judgment_id,
                match_id,
                side,
                decision,
                judge_model,
                explanation,
                raw_completion,
                source_log,
                sample_uuid,
            ),
        )
        if commit:
            conn.commit()

    def match_status_counts(self) -> dict[str, int]:
        """Count matches grouped by status."""
        conn = self.connection()
        rows = conn.execute(
            """
            SELECT status, COUNT(*) AS count
            FROM matches
            GROUP BY status
            """
        ).fetchall()
        return {str(row["status"]): int(row["count"]) for row in rows}

    def matches_for_status(self, statuses: list[str]) -> list[Row]:
        """Load all matches for the provided statuses."""
        if len(statuses) == 0:
            return []
        conn = self.connection()
        placeholders = ", ".join(["?"] * len(statuses))
        return list(
            conn.execute(
                f"""
                SELECT *
                FROM matches
                WHERE status IN ({placeholders})
                ORDER BY round_index, match_id
                """,
                statuses,
            ).fetchall()
        )

    def load_model_ratings(self) -> dict[str, ModelRating]:
        """Load current ratings keyed by model_id."""
        conn = self.connection()
        rows = conn.execute(
            """
            SELECT model_id, mu, sigma, games, wins, losses, ties
            FROM ratings
            """
        ).fetchall()
        return {
            str(row["model_id"]): ModelRating(
                model_id=str(row["model_id"]),
                mu=float(row["mu"]),
                sigma=float(row["sigma"]),
                games=int(row["games"]),
                wins=int(row["wins"]),
                losses=int(row["losses"]),
                ties=int(row["ties"]),
            )
            for row in rows
        }

    def upsert_model_rating(self, rating: ModelRating, *, commit: bool = True) -> None:
        """Insert or update one rating row."""
        conn = self.connection()
        conn.execute(
            """
            INSERT INTO ratings (
              model_id, mu, sigma, games, wins, losses, ties, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(model_id) DO UPDATE SET
              mu = excluded.mu,
              sigma = excluded.sigma,
              games = excluded.games,
              wins = excluded.wins,
              losses = excluded.losses,
              ties = excluded.ties,
              updated_at = CURRENT_TIMESTAMP
            """,
            (
                rating.model_id,
                rating.mu,
                rating.sigma,
                rating.games,
                rating.wins,
                rating.losses,
                rating.ties,
            ),
        )
        if commit:
            conn.commit()

    def next_ratings_history_step(self) -> int:
        """Get the next ratings history step id."""
        conn = self.connection()
        cursor = conn.execute("SELECT COALESCE(MAX(step_id), 0) + 1 AS step FROM ratings_history")
        row = cursor.fetchone()
        return int(row["step"])

    def append_ratings_history(
        self, step_id: int, ratings: dict[str, ModelRating], *, commit: bool = True
    ) -> None:
        """Append ratings snapshot for a given step."""
        conn = self.connection()
        for rating in ratings.values():
            conn.execute(
                """
                INSERT INTO ratings_history (step_id, model_id, mu, sigma, games)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(step_id, model_id) DO UPDATE SET
                  mu = excluded.mu,
                  sigma = excluded.sigma,
                  games = excluded.games
                """,
                (step_id, rating.model_id, rating.mu, rating.sigma, rating.games),
            )
        if commit:
            conn.commit()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        conn = self.connection()
        conn.execute("BEGIN")
        try:
            yield
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _apply_migrations(self) -> None:
        conn = self.connection()
        current_version = self.schema_version
        if current_version > SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version {current_version}; expected <= {SCHEMA_VERSION}"
            )

        for version in range(current_version + 1, SCHEMA_VERSION + 1):
            migration_sql = MIGRATIONS.get(version)
            if migration_sql is None:
                raise ValueError(f"No migration registered for schema version {version}")
            conn.executescript(migration_sql)
            conn.execute(f"PRAGMA user_version = {version}")
        conn.commit()


def initialize_tournament_store(
    config: TournamentConfig | Mapping[str, Any] | str | Path,
) -> Path:
    """Initialize tournament state from config and return database path."""
    parsed = load_tournament_config(config)
    if parsed.state_dir is None:
        raise ValueError("state_dir is required")

    with TournamentStore(parsed.state_dir) as store:
        store.initialize_from_config(parsed)
        return store.db_path
