"""
========================================================
player_db.py — Player Session History Database (V2)
========================================================

WHAT THIS FILE DOES:
--------------------
Provides a lightweight SQLite-based persistence layer for
tracking player performance across multiple sessions over time.

This enables:
  - Weekly improvement trend charts
  - Shot consistency % tracking
  - Injury risk history alerts
  - Personal record tracking

DATABASE SCHEMA:
  Table: players       — (id, name, created_at)
  Table: sessions      — (id, player_id, timestamp, overall_score, 
                          balance_score, stability_score, power_score,
                          timing_score, shot_type, video_filename)
  Table: session_feedback — (id, session_id, feedback_message, is_injury_risk)

AUTHOR: Cricket Biomechanics Analyzer V2
"""

import sqlite3
import os
from datetime import datetime
from pathlib import Path


DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'player_history.db')


class PlayerDatabase:
    """
    SQLite interface for persistent player session history.
    """

    def __init__(self, db_path: str = DB_PATH):
        self._db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_db()

    def _connect(self):
        """Open a thread-safe database connection."""
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Return dict-like rows
        return conn

    def _init_db(self):
        """Create tables if they don't already exist."""
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS players (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    name        TEXT    NOT NULL UNIQUE,
                    created_at  TEXT    NOT NULL DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id        INTEGER NOT NULL,
                    timestamp        TEXT    NOT NULL DEFAULT (datetime('now')),
                    overall_score    INTEGER,
                    balance_score    INTEGER,
                    stability_score  INTEGER,
                    power_score      INTEGER,
                    timing_score     INTEGER,
                    shot_type        TEXT,
                    swing_phase      TEXT,
                    video_filename   TEXT,
                    FOREIGN KEY (player_id) REFERENCES players(id)
                );

                CREATE TABLE IF NOT EXISTS session_feedback (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id       INTEGER NOT NULL,
                    feedback_message TEXT,
                    is_injury_risk   INTEGER DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sessions(id)
                );
            """)

    # ── Player Management ─────────────────────────────────────

    def get_or_create_player(self, name: str) -> int:
        """Returns player ID, creating the player if they don't exist."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM players WHERE name=?", (name,)
            ).fetchone()
            if row:
                return row['id']
            cur = conn.execute(
                "INSERT INTO players (name) VALUES (?)", (name,)
            )
            return cur.lastrowid

    def list_players(self) -> list:
        """Returns a list of all player names."""
        with self._connect() as conn:
            rows = conn.execute("SELECT name FROM players ORDER BY name").fetchall()
            return [r['name'] for r in rows]

    # ── Session Recording ─────────────────────────────────────

    def save_session(self, player_name: str, scores: dict,
                     shot_type: str = "Unknown",
                     swing_phase: str = "Unknown",
                     video_filename: str = "",
                     feedback: list = None,
                     injury_flags: list = None) -> int:
        """
        Persist a completed analysis session to the database.

        Args:
            player_name   (str):  Player's display name.
            scores        (dict): Score dict from BiomechanicsScorer.aggregate_session_scores().
            shot_type     (str):  Most common shot type detected.
            swing_phase   (str):  Most common swing phase detected.
            video_filename (str): Original uploaded video filename.
            feedback      (list): Coaching feedback messages.
            injury_flags  (list): Injury risk warnings.

        Returns:
            int: The new session ID.
        """
        player_id = self.get_or_create_player(player_name)
        feedback   = feedback   or []
        injury_flags = injury_flags or []

        with self._connect() as conn:
            cur = conn.execute("""
                INSERT INTO sessions
                    (player_id, overall_score, balance_score, stability_score,
                     power_score, timing_score, shot_type, swing_phase, video_filename)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                player_id,
                scores.get('overall_score',   0),
                scores.get('balance_score',   0),
                scores.get('stability_score', 0),
                scores.get('power_score',     0),
                scores.get('timing_score',    0),
                shot_type,
                swing_phase,
                video_filename,
            ))
            session_id = cur.lastrowid

            # Save feedback messages
            for msg in feedback:
                conn.execute("""
                    INSERT INTO session_feedback (session_id, feedback_message, is_injury_risk)
                    VALUES (?, ?, 0)
                """, (session_id, msg))

            for msg in injury_flags:
                conn.execute("""
                    INSERT INTO session_feedback (session_id, feedback_message, is_injury_risk)
                    VALUES (?, ?, 1)
                """, (session_id, msg))

            return session_id

    # ── Progress & Analytics Queries ─────────────────────────

    def get_player_history(self, player_name: str, limit: int = 30) -> list:
        """
        Returns a player's session history as a list of dicts.

        Args:
            player_name (str): Player name.
            limit (int):       Max number of sessions to return.

        Returns:
            list[dict]: Sessions ordered by most recent first.
        """
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT s.*
                FROM sessions s
                JOIN players p ON p.id = s.player_id
                WHERE p.name = ?
                ORDER BY s.timestamp DESC
                LIMIT ?
            """, (player_name, limit)).fetchall()
            return [dict(r) for r in rows]

    def get_player_trend(self, player_name: str) -> dict:
        """
        Returns improvement trends: lists of (date, score) for each dimension.
        Perfect for plotting time-series charts.
        """
        history = self.get_player_history(player_name, limit=30)
        if not history:
            return {}

        # Reverse so we go chronologically
        history = list(reversed(history))

        return {
            'timestamps':       [h['timestamp'][:10] for h in history],  # Date only
            'overall_score':    [h['overall_score']    for h in history],
            'balance_score':    [h['balance_score']    for h in history],
            'stability_score':  [h['stability_score']  for h in history],
            'power_score':      [h['power_score']      for h in history],
            'timing_score':     [h['timing_score']     for h in history],
        }

    def get_radar_data(self, player_name: str) -> dict:
        """
        Returns the last session's scores formatted for a radar/spider chart.
        """
        history = self.get_player_history(player_name, limit=1)
        if not history:
            return {}
        latest = history[0]
        return {
            'categories': ['Balance', 'Stability', 'Power', 'Timing'],
            'scores': [
                latest['balance_score'],
                latest['stability_score'],
                latest['power_score'],
                latest['timing_score'],
            ]
        }

    def get_personal_best(self, player_name: str) -> dict:
        """Returns the player's all-time best scores per dimension."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT
                    MAX(overall_score)    AS best_overall,
                    MAX(balance_score)    AS best_balance,
                    MAX(stability_score)  AS best_stability,
                    MAX(power_score)      AS best_power,
                    MAX(timing_score)     AS best_timing,
                    COUNT(*)              AS total_sessions
                FROM sessions s
                JOIN players p ON p.id = s.player_id
                WHERE p.name = ?
            """, (player_name,)).fetchone()
            if row:
                return dict(row)
            return {}

    def get_shot_distribution(self, player_name: str) -> dict:
        """Returns count of each shot type played by this player."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT shot_type, COUNT(*) as count
                FROM sessions s
                JOIN players p ON p.id = s.player_id
                WHERE p.name = ?
                GROUP BY shot_type
            """, (player_name,)).fetchall()
            return {r['shot_type']: r['count'] for r in rows}
