#!/usr/bin/env python3
"""ChangeLogメモをSQLite FTS5で検索するサーバー。"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import ipaddress
import logging
import math
import re
import sqlite3
import struct
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, urlencode, urlsplit


POLL_INTERVAL_SECONDS = 2.0
MAX_LIMIT = 100
LOOPBACK_HOST = "127.0.0.1"
LOGGER = logging.getLogger("clms")
DEFAULT_SEARCH_MODE = "hybrid"
SEARCH_MODES = ("hybrid", "bm25")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RRF_K = 60
SEMANTIC_TOP_K = 100
MIN_SEMANTIC_SIMILARITY = 0.70

HEADER_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<author>.*?)\s+<(?P<mail>[^<>]*)>\s*$"
)
ENTRY_RE = re.compile(r"^\t\*\s*(?P<content>.*)$")
URL_RE = re.compile(r"(?i)(?<![\w@])(?:https?://|www\.)[^\s<>\"']+")
TRAILING_URL_PUNCTUATION = ".,!?;:。、，．！？；：…」』》〉»”’"
URL_BRACKET_PAIRS = ((")", "("), ("]", "["), ("}", "{"), ("）", "（"), ("】", "【"))


class MemoFormatError(ValueError):
    """ChangeLogメモの書式エラー。"""


@dataclass
class EntryRecord:
    source_key: str
    tags: str
    title: str
    body: str
    sha1: str


@dataclass
class DayRecord:
    date: str
    author_name: str
    author_mail: str
    sha1: str
    entries: list[EntryRecord] = field(default_factory=list)


@dataclass
class SearchPage:
    rows: list[sqlite3.Row | dict[str, object]]
    total: int
    page: int
    limit: int
    mode: str = "bm25"


class Embedder(Protocol):
    model_id: str

    def encode(self, texts: list[str]) -> list[list[float]]:
        """複数の文章を正規化済みEmbeddingに変換する。"""


class EmbeddingError(RuntimeError):
    """Embeddingモデルの読み込みまたは推論に関するエラー。"""


class SentenceTransformerEmbedder:
    """sentence_transformersを遅延読み込みするEmbedding生成器。"""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_id = model_name
        self._model = None
        self._lock = threading.Lock()

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise EmbeddingError(
                        "ハイブリッド検索にはsentence-transformersが必要です。"
                        "python -m pip install -r requirements-embedding.txt を実行してください。"
                    ) from exc
                try:
                    self._model = SentenceTransformer(self.model_id)
                except Exception as exc:
                    raise EmbeddingError(
                        f"Embeddingモデルを読み込めません: {self.model_id}: {exc}"
                    ) from exc
            try:
                vectors = self._model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except Exception as exc:
                raise EmbeddingError(f"Embeddingを生成できません: {exc}") from exc
        return [[float(value) for value in vector] for vector in vectors]


def _sha1(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def _entry_digest(tags: str, title: str, body: str) -> str:
    return _sha1("\0".join((tags, title, body)))


def parse_exclude_tags(value: str) -> set[str]:
    return {tag.strip().casefold() for tag in value.split(",") if tag.strip()}


def _has_excluded_tag(tags: str, exclude_tags: set[str]) -> bool:
    return any(tag.strip().casefold() in exclude_tags for tag in tags.split(",") if tag.strip())


def parse_source(text: str) -> list[DayRecord]:
    """メモ本文を解析し、下から数えたエントリ位置キーを割り当てる。"""
    days: list[DayRecord] = []
    day_by_date: dict[str, DayRecord] = {}
    current_day: DayRecord | None = None
    current_entry: dict[str, object] | None = None
    body_lines: list[str] = []
    pending_blank_lines = 0

    def finish_entry() -> None:
        nonlocal current_entry, body_lines
        if current_entry is None:
            return
        assert current_day is not None
        tags = str(current_entry["tags"])
        title = str(current_entry["title"])
        body = "\n".join(body_lines)
        current_day.entries.append(
            EntryRecord("", tags, title, body, _entry_digest(tags, title, body))
        )
        current_entry = None
        body_lines = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.removesuffix("\r")
        if not line.strip() and not line.startswith("\t"):
            pending_blank_lines += 1
            continue

        header_match = HEADER_RE.fullmatch(line)
        if header_match:
            finish_entry()
            pending_blank_lines = 0
            date_text = header_match.group("date")
            try:
                dt.date.fromisoformat(date_text)
            except ValueError as exc:
                raise MemoFormatError(f"{line_number}行目: 不正な日付です: {date_text}") from exc
            if date_text in day_by_date:
                raise MemoFormatError(f"{line_number}行目: 日付が重複しています: {date_text}")
            author_name = header_match.group("author").strip()
            author_mail = header_match.group("mail").strip()
            day = DayRecord(
                date_text,
                author_name,
                author_mail,
                _sha1("\0".join((date_text, author_name, author_mail))),
            )
            days.append(day)
            day_by_date[date_text] = day
            current_day = day
            continue

        entry_match = ENTRY_RE.fullmatch(line)
        if entry_match:
            if current_day is None:
                raise MemoFormatError(f"{line_number}行目: 日付ヘッダーより前にエントリがあります")
            finish_entry()
            pending_blank_lines = 0
            content = entry_match.group("content").strip()
            if ":" in content:
                raw_tags, title = content.split(":", 1)
                raw_tags = raw_tags.strip()
                tag_list = [tag.strip() for tag in raw_tags.split(",")] if raw_tags else []
                if any(not tag for tag in tag_list):
                    raise MemoFormatError(f"{line_number}行目: タグの指定が不正です")
                tags = ", ".join(tag_list)
                title = title.strip()
            else:
                tags = ""
                title = content
            current_entry = {
                "tags": tags,
                "title": title,
            }
            body_lines = []
            continue

        if line.startswith("\t"):
            if current_entry is None:
                raise MemoFormatError(f"{line_number}行目: エントリ本文の位置が不正です")
            body_lines.extend([""] * pending_blank_lines)
            pending_blank_lines = 0
            body_lines.append(line[1:])
            continue

        raise MemoFormatError(f"{line_number}行目: 日付ヘッダーまたはタブ始まりのエントリが必要です")

    finish_entry()

    for day in days:
        entry_count = len(day.entries)
        for index, entry in enumerate(day.entries):
            entry.source_key = f"{day.date}_{entry_count - index:03d}"

    return days


def _connect(dbfile: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(dbfile), timeout=10.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 10000")
    return connection


def initialize_database(dbfile: str | Path) -> None:
    """テーブル、FTS5インデックス、同期トリガーを用意する。"""
    connection = _connect(dbfile)
    try:
        fts_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'entry_fts'"
        ).fetchone() is not None
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS day_header (
                id INTEGER PRIMARY KEY,
                date TEXT NOT NULL UNIQUE,
                author_name TEXT NOT NULL,
                author_mail TEXT NOT NULL,
                sha1 TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS entry (
                id INTEGER PRIMARY KEY,
                day_id INTEGER NOT NULL REFERENCES day_header(id) ON DELETE CASCADE,
                source_key TEXT NOT NULL UNIQUE,
                tags TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                sha1 TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS entry_embedding (
                entry_id INTEGER PRIMARY KEY REFERENCES entry(id) ON DELETE CASCADE,
                content_sha1 TEXT NOT NULL,
                model_id TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                vector BLOB NOT NULL
            );

            CREATE INDEX IF NOT EXISTS entry_embedding_model_idx
                ON entry_embedding(model_id);

            CREATE VIRTUAL TABLE IF NOT EXISTS entry_fts
            USING fts5(
                tags,
                title,
                body,
                content='entry',
                content_rowid='id',
                tokenize='trigram'
            );

            CREATE TRIGGER IF NOT EXISTS entry_ai AFTER INSERT ON entry BEGIN
                INSERT INTO entry_fts(rowid, tags, title, body)
                VALUES (new.id, new.tags, new.title, new.body);
            END;

            CREATE TRIGGER IF NOT EXISTS entry_ad AFTER DELETE ON entry BEGIN
                INSERT INTO entry_fts(entry_fts, rowid, tags, title, body)
                VALUES ('delete', old.id, old.tags, old.title, old.body);
            END;

            CREATE TRIGGER IF NOT EXISTS entry_au AFTER UPDATE ON entry BEGIN
                INSERT INTO entry_fts(entry_fts, rowid, tags, title, body)
                VALUES ('delete', old.id, old.tags, old.title, old.body);
                INSERT INTO entry_fts(rowid, tags, title, body)
                VALUES (new.id, new.tags, new.title, new.body);
            END;
            """
        )
        if not fts_exists:
            connection.execute("INSERT INTO entry_fts(entry_fts) VALUES ('rebuild')")
        connection.commit()
    except sqlite3.OperationalError as exc:
        connection.rollback()
        if "fts5" in str(exc).lower() or "trigram" in str(exc).lower():
            raise RuntimeError(
                "このPythonのSQLiteはFTS5 trigramに対応していません。"
                "FTS5とtrigram tokenizer対応のSQLiteを使用してください。"
            ) from exc
        raise
    finally:
        connection.close()


def _embedding_text(tags: str, title: str, body: str) -> str:
    return f"タイトル: {title}\nタグ: {tags}\n本文:\n{body}".strip()


def _normalize_vector(vector: list[float]) -> list[float]:
    if not vector or any(not math.isfinite(value) for value in vector):
        raise EmbeddingError("Embeddingの次元または値が不正です")
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        raise EmbeddingError("ゼロベクトルはEmbeddingとして保存できません")
    return [value / norm for value in vector]


def _pack_vector(vector: list[float]) -> tuple[int, bytes]:
    normalized = _normalize_vector(vector)
    return len(normalized), struct.pack(f"<{len(normalized)}f", *normalized)


def _unpack_vector(blob: bytes, dimensions: int) -> list[float]:
    if dimensions < 1 or len(blob) != dimensions * 4:
        raise EmbeddingError("データベース内のEmbedding形式が不正です")
    return list(struct.unpack(f"<{dimensions}f", blob))


def _cached_embedding_vectors(dbfile: str | Path, model_id: str) -> dict[str, tuple[int, bytes]]:
    connection = _connect(dbfile)
    try:
        rows = connection.execute(
            """
            SELECT e.sha1, ee.content_sha1, ee.model_id, ee.dimensions, ee.vector
            FROM entry AS e
            JOIN entry_embedding AS ee ON ee.entry_id = e.id
            WHERE ee.model_id = ?
            """,
            (model_id,),
        )
        cached: dict[str, tuple[int, bytes]] = {}
        for row in rows:
            if row["sha1"] != row["content_sha1"]:
                continue
            dimensions = row["dimensions"]
            vector = row["vector"]
            if dimensions > 0 and len(vector) == dimensions * 4:
                cached[row["sha1"]] = (dimensions, vector)
        return cached
    finally:
        connection.close()


def _prepare_embedding_vectors(
    dbfile: str | Path,
    entries: list[EntryRecord],
    embedder: Embedder,
) -> dict[str, tuple[int, bytes]]:
    vectors = _cached_embedding_vectors(dbfile, embedder.model_id)
    missing: dict[str, EntryRecord] = {}
    for entry in entries:
        if entry.sha1 not in vectors:
            missing.setdefault(entry.sha1, entry)
    if not missing:
        return vectors

    texts = [
        _embedding_text(entry.tags, entry.title, entry.body)
        for entry in missing.values()
    ]
    encoded = embedder.encode(texts)
    if len(encoded) != len(texts):
        raise EmbeddingError("Embeddingモデルが返した件数が一致しません")
    for digest, vector in zip(missing, encoded):
        vectors[digest] = _pack_vector(vector)
    return vectors


def _store_embedding(
    connection: sqlite3.Connection,
    entry_id: int,
    digest: str,
    model_id: str,
    dimensions: int,
    vector: bytes,
) -> None:
    connection.execute(
        """
        INSERT INTO entry_embedding(entry_id, content_sha1, model_id, dimensions, vector)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(entry_id) DO UPDATE SET
            content_sha1 = excluded.content_sha1,
            model_id = excluded.model_id,
            dimensions = excluded.dimensions,
            vector = excluded.vector
        WHERE entry_embedding.content_sha1 != excluded.content_sha1
           OR entry_embedding.model_id != excluded.model_id
           OR entry_embedding.dimensions != excluded.dimensions
        """,
        (entry_id, digest, model_id, dimensions, vector),
    )


def _desired_entries_for_day(
    day: DayRecord,
    existing: dict[str, str],
    excluded: set[str],
) -> dict[str, EntryRecord]:
    all_desired = {entry.source_key: entry for entry in day.entries}
    existing_hashes = set(existing.values())
    excluded_new_keys: set[str] = set()
    if excluded:
        for entry in day.entries:
            if not _has_excluded_tag(entry.tags, excluded):
                continue
            previous_digest = existing.get(entry.source_key)
            if previous_digest is None:
                if entry.sha1 not in existing_hashes:
                    excluded_new_keys.add(entry.source_key)
            elif previous_digest != entry.sha1:
                previous_digest_moved = any(
                    other_key != entry.source_key and other.sha1 == previous_digest
                    for other_key, other in all_desired.items()
                )
                if previous_digest_moved:
                    excluded_new_keys.add(entry.source_key)
    return {
        source_key: entry
        for source_key, entry in all_desired.items()
        if source_key not in excluded_new_keys
    }


def sync_database(
    dbfile: str | Path,
    days: list[DayRecord],
    exclude_tags: set[str] | None = None,
    embedder: Embedder | None = None,
) -> tuple[int, int, int]:
    """解析済みデータを差分同期し、(追加, 更新, 削除)件数を返す。"""
    added = updated = deleted = 0
    excluded = exclude_tags or set()
    desired_dates = {day.date for day in days}
    embedding_vectors: dict[str, tuple[int, bytes]] = {}
    if embedder is not None:
        read_connection = _connect(dbfile)
        try:
            embedding_entries: list[EntryRecord] = []
            for day in days:
                existing = {
                    row["source_key"]: row["sha1"]
                    for row in read_connection.execute(
                        """
                        SELECT e.source_key, e.sha1
                        FROM entry AS e
                        JOIN day_header AS d ON d.id = e.day_id
                        WHERE d.date = ?
                        """,
                        (day.date,),
                    )
                }
                embedding_entries.extend(
                    _desired_entries_for_day(day, existing, excluded).values()
                )
        finally:
            read_connection.close()
        embedding_vectors = _prepare_embedding_vectors(dbfile, embedding_entries, embedder)

    connection = _connect(dbfile)
    try:
        connection.execute("BEGIN IMMEDIATE")
        for day in days:
            connection.execute(
                """
                INSERT INTO day_header(date, author_name, author_mail, sha1)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    author_name = excluded.author_name,
                    author_mail = excluded.author_mail,
                    sha1 = excluded.sha1
                """,
                (day.date, day.author_name, day.author_mail, day.sha1),
            )
            day_id = connection.execute(
                "SELECT id FROM day_header WHERE date = ?", (day.date,)
            ).fetchone()["id"]

            existing = {
                row["source_key"]: row["sha1"]
                for row in connection.execute(
                    "SELECT source_key, sha1 FROM entry WHERE day_id = ?", (day_id,)
                )
            }
            desired = _desired_entries_for_day(day, existing, excluded)

            # 位置キーがずれた場合も一意制約に抵触しないよう、差分行を先に削除する。
            keys_to_remove = [
                source_key
                for source_key, digest in existing.items()
                if source_key not in desired or desired[source_key].sha1 != digest
            ]
            for source_key in keys_to_remove:
                connection.execute("DELETE FROM entry WHERE source_key = ?", (source_key,))
                deleted += 1

            for entry in desired.values():
                previous_digest = existing.get(entry.source_key)
                if previous_digest == entry.sha1:
                    continue
                connection.execute(
                    """
                    INSERT INTO entry(day_id, source_key, tags, title, body, sha1)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (day_id, entry.source_key, entry.tags, entry.title, entry.body, entry.sha1),
                )
                if previous_digest is None:
                    added += 1
                else:
                    updated += 1

            if embedder is not None:
                for entry in desired.values():
                    row = connection.execute(
                        "SELECT id FROM entry WHERE source_key = ? AND sha1 = ?",
                        (entry.source_key, entry.sha1),
                    ).fetchone()
                    if row is None:
                        continue
                    dimensions, vector = embedding_vectors[entry.sha1]
                    _store_embedding(
                        connection,
                        row["id"],
                        entry.sha1,
                        embedder.model_id,
                        dimensions,
                        vector,
                    )

        stale_dates = [
            row["date"]
            for row in connection.execute("SELECT date FROM day_header")
            if row["date"] not in desired_dates
        ]
        for date_text in stale_dates:
            day_id = connection.execute(
                "SELECT id FROM day_header WHERE date = ?", (date_text,)
            ).fetchone()["id"]
            deleted += connection.execute(
                "SELECT COUNT(*) FROM entry WHERE day_id = ?", (day_id,)
            ).fetchone()[0]
            connection.execute("DELETE FROM day_header WHERE id = ?", (day_id,))

        connection.commit()
        return added, updated, deleted
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _short_query_rank(tags: str, title: str, body: str, *queries: str) -> int:
    normalized_tags = tags.casefold()
    normalized_title = title.casefold()
    normalized_body = body.casefold()
    return sum(
        normalized_tags.count(query.casefold()) * 5
        + normalized_title.count(query.casefold()) * 4
        + normalized_body.count(query.casefold())
        for query in queries
    )


def _like_condition(term: str) -> tuple[str, tuple[str, str, str]]:
    escaped = term.replace("!", "!!").replace("%", "!%").replace("_", "!_")
    pattern = f"%{escaped}%"
    condition = "(e.tags LIKE ? ESCAPE '!' OR e.title LIKE ? ESCAPE '!' OR e.body LIKE ? ESCAPE '!')"
    return condition, (pattern, pattern, pattern)


def _bm25_results(connection: sqlite3.Connection, query: str) -> list[sqlite3.Row]:
    terms = query.split()
    long_terms = [term for term in terms if len(term) >= 3]
    short_terms = [term for term in terms if len(term) < 3]
    if not terms:
        return []

    short_conditions_and_params = [_like_condition(term) for term in short_terms]
    short_where = " AND ".join(condition for condition, _ in short_conditions_and_params)
    short_params = tuple(
        value
        for _, parameters in short_conditions_and_params
        for value in parameters
    )

    if long_terms:
        # 各語を引用してFTS5演算子を無効化し、すべての語をAND検索する。
        fts_terms = ['"' + term.replace('"', '""') + '"' for term in long_terms]
        fts_query = " AND ".join(fts_terms)
        where_clause = "entry_fts MATCH ?"
        if short_where:
            where_clause += " AND " + short_where
        parameters = (fts_query, *short_params)
        return connection.execute(
            f"""
            SELECT e.id AS entry_id, e.source_key, d.date, d.author_name, d.author_mail,
                   e.tags, e.title, e.body, bm25(entry_fts, 5.0, 8.0, 1.0) AS score
            FROM entry_fts
            JOIN entry AS e ON e.id = entry_fts.rowid
            JOIN day_header AS d ON d.id = e.day_id
            WHERE {where_clause}
            ORDER BY score ASC, d.date DESC, e.source_key DESC
            """,
            parameters,
        ).fetchall()

    short_where = " AND ".join(condition for condition, _ in short_conditions_and_params)
    score_placeholders = ", ".join("?" for _ in short_terms)
    connection.create_function("short_query_rank", -1, _short_query_rank)
    return connection.execute(
        f"""
        SELECT e.id AS entry_id, e.source_key, d.date, d.author_name, d.author_mail,
               e.tags, e.title, e.body,
               short_query_rank(e.tags, e.title, e.body, {score_placeholders}) AS score
        FROM entry AS e
        JOIN day_header AS d ON d.id = e.day_id
        WHERE {short_where}
        ORDER BY score DESC, d.date DESC, e.source_key DESC
        """,
        (*short_terms, *short_params),
    ).fetchall()


def _ensure_database_embeddings(dbfile: str | Path, embedder: Embedder) -> None:
    connection = _connect(dbfile)
    try:
        rows = connection.execute(
            """
            SELECT e.id, e.source_key, e.sha1, e.tags, e.title, e.body,
                   ee.content_sha1, ee.model_id, ee.dimensions, ee.vector
            FROM entry AS e
            LEFT JOIN entry_embedding AS ee ON ee.entry_id = e.id
            """
        ).fetchall()
    finally:
        connection.close()

    missing: list[tuple[int, str]] = []
    records: list[EntryRecord] = []
    for row in rows:
        vector_is_current = (
            row["content_sha1"] == row["sha1"]
            and row["model_id"] == embedder.model_id
            and row["dimensions"] is not None
            and row["dimensions"] > 0
            and len(row["vector"]) == row["dimensions"] * 4
        )
        if vector_is_current:
            continue
        missing.append((row["id"], row["sha1"]))
        records.append(
            EntryRecord(
                row["source_key"],
                row["tags"],
                row["title"],
                row["body"],
                row["sha1"],
            )
        )
    if not missing:
        return

    vectors = _prepare_embedding_vectors(dbfile, records, embedder)
    connection = _connect(dbfile)
    try:
        connection.execute("BEGIN IMMEDIATE")
        for entry_id, digest in missing:
            current = connection.execute(
                "SELECT sha1 FROM entry WHERE id = ?", (entry_id,)
            ).fetchone()
            if current is None or current["sha1"] != digest:
                continue
            dimensions, vector = vectors[digest]
            _store_embedding(
                connection, entry_id, digest, embedder.model_id, dimensions, vector
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _semantic_results(
    connection: sqlite3.Connection,
    query: str,
    embedder: Embedder,
) -> list[tuple[sqlite3.Row, float]]:
    query_vectors = embedder.encode([query])
    if len(query_vectors) != 1:
        raise EmbeddingError("Embeddingモデルが検索語を変換できませんでした")
    query_vector = _normalize_vector(query_vectors[0])
    rows = connection.execute(
        """
        SELECT e.id AS entry_id, e.source_key, d.date, d.author_name, d.author_mail,
               e.tags, e.title, e.body, ee.dimensions, ee.vector
        FROM entry AS e
        JOIN entry_embedding AS ee ON ee.entry_id = e.id
        JOIN day_header AS d ON d.id = e.day_id
        WHERE ee.model_id = ? AND ee.content_sha1 = e.sha1
        """,
        (embedder.model_id,),
    )
    scored: list[tuple[sqlite3.Row, float]] = []
    for row in rows:
        if row["dimensions"] != len(query_vector):
            continue
        vector = _unpack_vector(row["vector"], row["dimensions"])
        similarity = sum(left * right for left, right in zip(query_vector, vector))
        if similarity >= MIN_SEMANTIC_SIMILARITY:
            scored.append((row, similarity))
    scored.sort(
        key=lambda item: (
            item[1],
            item[0]["date"],
            item[0]["source_key"],
        ),
        reverse=True,
    )
    return scored[:SEMANTIC_TOP_K]


def search_database(
    dbfile: str | Path,
    query: str,
    page: int,
    limit: int,
    mode: str = "bm25",
    embedder: Embedder | None = None,
) -> SearchPage:
    if page < 1:
        raise ValueError("pageは1以上を指定してください")
    if limit < 1 or limit > MAX_LIMIT:
        raise ValueError(f"limitは1〜{MAX_LIMIT}を指定してください")
    if mode not in SEARCH_MODES:
        raise ValueError("modeはhybridまたはbm25を指定してください")
    row_offset = (page - 1) * limit
    if not query.strip():
        return SearchPage([], 0, page, limit, mode)

    if mode == "bm25":
        connection = _connect(dbfile)
        try:
            candidates = _bm25_results(connection, query)
        finally:
            connection.close()
        return SearchPage(
            candidates[row_offset:row_offset + limit],
            len(candidates),
            page,
            limit,
            mode,
        )

    if embedder is None:
        raise EmbeddingError("ハイブリッド検索用のEmbeddingモデルが設定されていません")
    _ensure_database_embeddings(dbfile, embedder)
    connection = _connect(dbfile)
    try:
        keyword_candidates = _bm25_results(connection, query)
        semantic_candidates = _semantic_results(connection, query, embedder)
    finally:
        connection.close()

    candidate_rows: dict[int, sqlite3.Row] = {}
    rrf_scores: dict[int, float] = {}
    for rank, row in enumerate(keyword_candidates, start=1):
        entry_id = row["entry_id"]
        candidate_rows[entry_id] = row
        rrf_scores[entry_id] = rrf_scores.get(entry_id, 0.0) + 1 / (RRF_K + rank)
    for rank, (row, _) in enumerate(semantic_candidates, start=1):
        entry_id = row["entry_id"]
        candidate_rows[entry_id] = row
        rrf_scores[entry_id] = rrf_scores.get(entry_id, 0.0) + 1 / (RRF_K + rank)

    ranked_ids = sorted(
        candidate_rows,
        key=lambda entry_id: (
            rrf_scores[entry_id],
            candidate_rows[entry_id]["date"],
            candidate_rows[entry_id]["source_key"],
        ),
        reverse=True,
    )
    ordered_rows = [dict(candidate_rows[entry_id]) for entry_id in ranked_ids]
    return SearchPage(
        ordered_rows[row_offset:row_offset + limit],
        len(ordered_rows),
        page,
        limit,
        mode,
    )


def _trim_url_punctuation(value: str) -> tuple[str, str]:
    end = len(value)
    while end and value[end - 1] in TRAILING_URL_PUNCTUATION:
        end -= 1
    candidate = value[:end]
    for closing, opening in URL_BRACKET_PAIRS:
        while candidate.endswith(closing) and candidate.count(closing) > candidate.count(opening):
            candidate = candidate[:-1]
            end -= 1
    return candidate, value[end:]


def linkify(text: str) -> str:
    """HTTP(S) URLとwww.表記を、安全にエスケープしてリンク化する。"""
    pieces: list[str] = []
    cursor = 0
    for match in URL_RE.finditer(text):
        pieces.append(html.escape(text[cursor:match.start()]))
        visible, suffix = _trim_url_punctuation(match.group(0))
        if not visible:
            pieces.append(html.escape(match.group(0)))
            cursor = match.end()
            continue
        href = visible if visible.lower().startswith(("http://", "https://")) else f"https://{visible}"
        parsed = urlsplit(href)
        if parsed.scheme in ("http", "https") and parsed.netloc:
            pieces.append(
                f'<a href="{html.escape(href, quote=True)}" rel="noopener noreferrer">'
                f"{html.escape(visible)}</a>"
            )
            pieces.append(html.escape(suffix))
        else:
            pieces.append(html.escape(match.group(0)))
        cursor = match.end()
    pieces.append(html.escape(text[cursor:]))
    return "".join(pieces)


def _pagination_link(
    query: str, page: int, limit: int, mode: str, label: str, relation: str
) -> str:
    target = "/search?" + urlencode(
        {"q": query, "page": page, "limit": limit, "mode": mode}
    )
    return (
        f'<a rel="{relation}" href="{html.escape(target, quote=True)}">'
        f"{html.escape(label)}</a>"
    )


def render_search_page(query: str, result: SearchPage) -> str:
    safe_query = html.escape(query, quote=True)
    parts = [
        "<!doctype html>",
        '<html lang="ja">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>ChangeLogメモ検索</title>",
        '<link rel="stylesheet" href="/style.css">',
        '<script src="/shortcuts.js" defer></script>',
        "</head>",
        "<body>",
        '<main class="container">',
        "<h1>ChangeLogメモ検索</h1>",
        '<form action="/search" method="get" class="search-form">',
        f'<input type="search" name="q" value="{safe_query}" aria-label="検索語">',
        '<input type="hidden" name="limit" value="{}">'.format(result.limit),
        '<label for="search-mode">検索方式</label>',
        '<select id="search-mode" name="mode">',
        '<option value="hybrid"{}>ハイブリッド</option>'.format(
            " selected" if result.mode == "hybrid" else ""
        ),
        '<option value="bm25"{}>BM25のみ</option>'.format(
            " selected" if result.mode == "bm25" else ""
        ),
        "</select>",
        '<button type="submit">検索</button>',
        "</form>",
    ]
    if any(len(term) < 3 for term in query.split()):
        parts.append('<p class="search-note">短い検索語は部分一致で検索しています。</p>')
    if query.strip():
        first_result = (result.page - 1) * result.limit + 1 if result.total else 0
        parts.append(
            f'<p class="result-count">{result.total}件中、{first_result}件目以降</p>'
        )
    if not result.rows and query:
        parts.append('<p class="empty">該当するメモはありません。</p>')

    for row in result.rows:
        author = row["author_name"]
        if row["author_mail"]:
            author = f"{author} <{row['author_mail']}>"
        parts.extend(
            [
                '<article class="entry">',
                '<header class="entry-header">',
                f'<time datetime="{html.escape(row["date"], quote=True)}">{html.escape(row["date"])}</time>',
                f'<span class="author">{html.escape(author)}</span>',
                "</header>",
                f'<h2>{linkify(row["title"])}</h2>',
                f'<p class="tags">{linkify(row["tags"])}</p>',
                f'<div class="entry-body">{linkify(row["body"])}</div>',
                "</article>",
            ]
        )

    if result.total > result.limit:
        parts.append('<nav class="pagination" aria-label="ページ移動">')
        if result.page > 1:
            parts.append(
                _pagination_link(
                    query, result.page - 1, result.limit, result.mode, "前へ", "prev"
                )
            )
        last_page = (result.total + result.limit - 1) // result.limit
        parts.append(f"<span>{result.page} / {last_page}</span>")
        if result.page < last_page:
            parts.append(
                _pagination_link(
                    query, result.page + 1, result.limit, result.mode, "次へ", "next"
                )
            )
        parts.append("</nav>")
    parts.extend(["</main>", "</body>", "</html>"])
    return "\n".join(parts)


def _read_source(source: Path) -> tuple[str, str]:
    content = source.read_bytes()
    digest = hashlib.sha1(content).hexdigest()
    return content.decode("utf-8-sig"), digest


def _is_loopback_address(address: str) -> bool:
    try:
        return ipaddress.ip_address(address).is_loopback
    except ValueError:
        return False


class SourceWatcher(threading.Thread):
    def __init__(
        self,
        source: Path,
        dbfile: Path,
        initial_digest: str,
        exclude_tags: set[str] | None = None,
        embedder: Embedder | None = None,
    ):
        super().__init__(name="clms-source-watcher", daemon=True)
        self.source = source
        self.dbfile = dbfile
        self.digest = initial_digest
        self.exclude_tags = exclude_tags or set()
        self.embedder = embedder
        self.pending_digest: str | None = None
        self.stop_event = threading.Event()

    def run(self) -> None:
        while not self.stop_event.wait(POLL_INTERVAL_SECONDS):
            try:
                text, digest = _read_source(self.source)
                if digest == self.digest:
                    self.pending_digest = None
                    continue
                if digest != self.pending_digest:
                    self.pending_digest = digest
                    continue
                days = parse_source(text)
                changes = sync_database(
                    self.dbfile, days, self.exclude_tags, self.embedder
                )
                self.digest = digest
                self.pending_digest = None
                LOGGER.info(
                    "ソースを同期しました (追加 %d、更新 %d、削除 %d)", *changes
                )
            except (OSError, UnicodeError, MemoFormatError, RuntimeError, sqlite3.Error) as exc:
                LOGGER.warning("ソース更新を反映できませんでした: %s", exc)

    def stop(self) -> None:
        self.stop_event.set()


def make_handler(
    dbfile: Path,
    cssfile: Path,
    default_mode: str = DEFAULT_SEARCH_MODE,
    embedder: Embedder | None = None,
) -> type[BaseHTTPRequestHandler]:
    class ClmsRequestHandler(BaseHTTPRequestHandler):
        server_version = "CLMS/1.0"

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; style-src 'self'; base-uri 'none'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if not _is_loopback_address(self.client_address[0]):
                self._send(403, "text/plain; charset=utf-8", "Forbidden".encode("utf-8"))
                return

            request = urlsplit(self.path)
            if request.path == "/style.css":
                try:
                    css = cssfile.read_bytes()
                except OSError:
                    self._send(404, "text/plain; charset=utf-8", "CSSが見つかりません。".encode("utf-8"))
                    return
                self._send(200, "text/css; charset=utf-8", css)
                return
            if request.path == "/shortcuts.js":
                try:
                    script = Path(__file__).with_name("clms.js").read_bytes()
                except OSError:
                    self._send(
                        404,
                        "text/plain; charset=utf-8",
                        "JavaScriptが見つかりません。".encode("utf-8"),
                    )
                    return
                self._send(200, "application/javascript; charset=utf-8", script)
                return
            if request.path not in ("/", "/search"):
                self._send(404, "text/plain; charset=utf-8", "Not Found".encode("utf-8"))
                return

            parameters = parse_qs(request.query, keep_blank_values=True, encoding="utf-8")
            query = parameters.get("q", [""])[0]
            mode = parameters.get("mode", [default_mode])[0]
            try:
                page = int(parameters.get("page", ["1"])[0])
                limit = int(parameters.get("limit", ["10"])[0])
                result = search_database(dbfile, query, page, limit, mode, embedder)
            except ValueError as exc:
                message = f"不正なリクエストです: {exc}"
                self._send(400, "text/plain; charset=utf-8", message.encode("utf-8"))
                return
            except EmbeddingError as exc:
                self._send(503, "text/plain; charset=utf-8", str(exc).encode("utf-8"))
                return
            except sqlite3.Error:
                LOGGER.exception("検索に失敗しました")
                self._send(500, "text/plain; charset=utf-8", "検索に失敗しました。".encode("utf-8"))
                return

            body = render_search_page(query, result).encode("utf-8")
            self._send(200, "text/html; charset=utf-8", body)

        def log_message(self, format_string: str, *args: object) -> None:
            LOGGER.info("%s - %s", self.address_string(), format_string % args)

    return ClmsRequestHandler


def _parse_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ポート番号は整数で指定してください") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("ポート番号は1〜65535で指定してください")
    return port


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ChangeLogメモをSQLite FTS5で検索するサーバー")
    parser.add_argument("--source", required=True, help="ChangeLogメモのテキストファイル")
    parser.add_argument("--dbfile", required=True, help="SQLiteデータベースファイル")
    parser.add_argument("--port", required=True, type=_parse_port, help="待ち受けポート番号")
    parser.add_argument(
        "--exclude-tags",
        default="",
        help="新規登録しないタグ (カンマ区切り)",
    )
    parser.add_argument(
        "--search-mode",
        choices=SEARCH_MODES,
        default=DEFAULT_SEARCH_MODE,
        help="検索方式の既定値 (hybrid または bm25)",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="sentence-transformersのモデル名またはローカルパス",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    source = Path(args.source).expanduser().resolve()
    dbfile = Path(args.dbfile).expanduser().resolve()
    exclude_tags = parse_exclude_tags(args.exclude_tags)
    embedder = SentenceTransformerEmbedder(args.embedding_model)
    if source == dbfile:
        parser.error("--sourceと--dbfileには異なるファイルを指定してください")
    if not source.is_file():
        parser.error(f"ソースファイルが見つかりません: {source}")

    try:
        text, digest = _read_source(source)
        days = parse_source(text)
        initialize_database(dbfile)
        changes = sync_database(
            dbfile,
            days,
            exclude_tags,
            embedder if args.search_mode == "hybrid" else None,
        )
    except (OSError, UnicodeError, MemoFormatError, RuntimeError, sqlite3.Error) as exc:
        parser.error(str(exc))
    LOGGER.info("初期同期しました (追加 %d、更新 %d、削除 %d)", *changes)

    cssfile = Path(__file__).with_name("clms.css")
    watcher = SourceWatcher(
        source,
        dbfile,
        digest,
        exclude_tags,
        embedder if args.search_mode == "hybrid" else None,
    )
    try:
        server = ThreadingHTTPServer(
            (LOOPBACK_HOST, args.port),
            make_handler(dbfile, cssfile, args.search_mode, embedder),
        )
    except OSError as exc:
        LOGGER.error(
            "%s:%d で待ち受けできません。ポートが使用中、または予約されている可能性があります: %s",
            LOOPBACK_HOST,
            args.port,
            exc,
        )
        return 1
    watcher.start()
    LOGGER.info("http://%s:%d/ で待ち受けています", LOOPBACK_HOST, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("サーバーを終了します")
    finally:
        server.server_close()
        watcher.stop()
        watcher.join(timeout=POLL_INTERVAL_SECONDS + 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
