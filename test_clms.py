import tempfile
import threading
import unittest
from unittest.mock import patch
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen
from http.server import ThreadingHTTPServer

import clms


SAMPLE = """2026-11-23 山田太郎 <yamada@example.com>
\t* 開発, Python : SQLiteの検索機能
\tFTS5でメモを検索する。https://example.com/docs.

\t* 日記 : 買い物
\t牛乳を買う

2026-11-22 山田太郎 <yamada@example.com>
\t* 読書 : 本の記録
\tSQLiteを読む
"""


class FakeEmbedder:
    model_id = "test-embedding-v1"

    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            if any(word in text for word in ("ランチ", "昼食", "食事")):
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])
        return vectors


class ParseTests(unittest.TestCase):
    def test_parse_assigns_bottom_up_keys_and_fields(self):
        days = clms.parse_source(SAMPLE)
        self.assertEqual([day.date for day in days], ["2026-11-23", "2026-11-22"])
        self.assertEqual(
            [entry.source_key for entry in days[0].entries],
            ["2026-11-23_002", "2026-11-23_001"],
        )
        self.assertEqual(days[0].entries[0].tags, "開発, Python")
        self.assertIn("https://example.com/docs.", days[0].entries[0].body)

    def test_rejects_entry_before_day_header(self):
        with self.assertRaises(clms.MemoFormatError):
            clms.parse_source("\t* tag : title\n\tbody\n")

    def test_tab_only_lines_are_preserved_as_empty_body_lines(self):
        source = (
            "2026-11-23 山田太郎 <yamada@example.com>\n"
            "\t* tag : title\n"
            "\t前半\n"
            "\t\n"
            "\t後半\n"
            "\t\n"
            "\n"
        )
        entry = clms.parse_source(source)[0].entries[0]
        self.assertEqual(entry.body, "前半\n\n後半\n")

    def test_tagless_entries_and_blank_lines_inside_body(self):
        source = (
            "2026-11-23 山田太郎 <yamada@example.com>\n"
            "\t* tag : tagged title\n"
            "\t本文の前半\n"
            "\n"
            "\t本文の後半: コロンも本文として扱う\n"
            "\n"
            "\t* aiueo\n"
            "\tタグなし本文\n"
        )
        entries = clms.parse_source(source)[0].entries
        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0].tags, "tag")
        self.assertEqual(entries[0].body, "本文の前半\n\n本文の後半: コロンも本文として扱う")
        self.assertEqual(entries[1].tags, "")
        self.assertEqual(entries[1].title, "aiueo")


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.dbfile = Path(self.tempdir.name) / "memo.sqlite3"
        clms.initialize_database(self.dbfile)

    def tearDown(self):
        self.tempdir.cleanup()

    def test_sync_search_and_short_query(self):
        days = clms.parse_source(SAMPLE)
        self.assertEqual(clms.sync_database(self.dbfile, days), (3, 0, 0))

        page = clms.search_database(self.dbfile, "SQLite", 1, 10)
        self.assertEqual(page.total, 2)
        self.assertEqual(page.rows[0]["date"], "2026-11-23")

        short_page = clms.search_database(self.dbfile, "本", 1, 10)
        self.assertEqual(short_page.total, 1)
        self.assertEqual(short_page.rows[0]["title"], "本の記録")

        and_page = clms.search_database(self.dbfile, "SQLite FTS5", 1, 10)
        self.assertEqual(and_page.total, 1)
        self.assertEqual(and_page.rows[0]["title"], "SQLiteの検索機能")

        mixed_length_page = clms.search_database(self.dbfile, "SQLite 本", 1, 10)
        self.assertEqual(mixed_length_page.total, 1)
        self.assertEqual(mixed_length_page.rows[0]["title"], "本の記録")

        no_and_match = clms.search_database(self.dbfile, "SQLite 買い物", 1, 10)
        self.assertEqual(no_and_match.total, 0)

        short_terms_page = clms.search_database(self.dbfile, "牛乳 買う", 1, 10)
        self.assertEqual(short_terms_page.total, 1)
        self.assertEqual(short_terms_page.rows[0]["title"], "買い物")

    def test_sync_updates_changed_position_and_removes_deleted_entries(self):
        original = clms.parse_source(SAMPLE)
        clms.sync_database(self.dbfile, original)
        changed = clms.parse_source(
            "2026-11-23 山田太郎 <yamada@example.com>\n"
            "\t* 開発 : 変更後のタイトル\n\t更新した本文\n"
        )
        added, updated, deleted = clms.sync_database(self.dbfile, changed)
        self.assertEqual((added, updated, deleted), (0, 1, 3))
        self.assertEqual(clms.search_database(self.dbfile, "変更後", 1, 10).total, 1)
        self.assertEqual(clms.search_database(self.dbfile, "SQLite", 1, 10).total, 0)

    def test_exclude_tags_only_skips_new_records(self):
        days = clms.parse_source(SAMPLE)
        excluded = clms.parse_exclude_tags(" Python, ")
        self.assertEqual(excluded, {"python"})
        self.assertEqual(clms.sync_database(self.dbfile, days, excluded), (2, 0, 0))
        self.assertEqual(clms.search_database(self.dbfile, "検索機能", 1, 10).total, 0)

        legacy_db = Path(self.tempdir.name) / "legacy.sqlite3"
        clms.initialize_database(legacy_db)
        clms.sync_database(legacy_db, days)
        self.assertEqual(clms.sync_database(legacy_db, days, excluded), (0, 0, 0))
        self.assertEqual(clms.search_database(legacy_db, "検索機能", 1, 10).total, 1)

        expanded_source = SAMPLE.replace(
            "\n\n2026-11-22",
            "\n\n\t* private : 対象外エントリ\n\t秘密の本文\n\n2026-11-22",
        )
        expanded_days = clms.parse_source(expanded_source)
        clms.sync_database(legacy_db, expanded_days, {"python", "private"})
        self.assertEqual(clms.search_database(legacy_db, "対象外", 1, 10).total, 0)
        self.assertEqual(clms.search_database(legacy_db, "検索機能", 1, 10).total, 1)

    def test_hybrid_search_finds_semantically_related_entry_and_reuses_vectors(self):
        source = (
            "2026-11-23 山田太郎 <yamada@example.com>\n"
            "\t* 日記 : ランチの支出\n"
            "\t食事代を記録した\n"
            "\t* 開発 : Python設定\n"
            "\t開発環境を整える\n"
        )
        days = clms.parse_source(source)
        embedder = FakeEmbedder()
        self.assertEqual(clms.sync_database(self.dbfile, days, embedder=embedder), (2, 0, 0))
        self.assertEqual(len(embedder.calls), 1)

        clms.sync_database(self.dbfile, days, embedder=embedder)
        self.assertEqual(len(embedder.calls), 1)

        keyword_page = clms.search_database(
            self.dbfile, "昼食代", 1, 10, mode="bm25", embedder=embedder
        )
        self.assertEqual(keyword_page.total, 0)

        hybrid_page = clms.search_database(
            self.dbfile, "昼食代", 1, 10, mode="hybrid", embedder=embedder
        )
        self.assertEqual(hybrid_page.total, 1)
        self.assertEqual(hybrid_page.rows[0]["title"], "ランチの支出")
        self.assertEqual(hybrid_page.mode, "hybrid")
        self.assertEqual(len(embedder.calls), 2)


class HtmlTests(unittest.TestCase):
    def test_linkify_escapes_html_and_links_supported_urls(self):
        rendered = clms.linkify('<script>alert(1)</script> https://example.com/path. www.example.org')
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn('href="https://example.com/path"', rendered)
        self.assertIn('href="https://www.example.org"', rendered)
        self.assertTrue(rendered.endswith("www.example.org</a>"))

    def test_linkify_keeps_balanced_parentheses_in_url(self):
        rendered = clms.linkify("https://example.com/a(b).")
        self.assertIn('href="https://example.com/a(b)"', rendered)
        self.assertTrue(rendered.endswith("</a>."))

    def test_render_search_page_separates_code_blocks_and_shows_language(self):
        result = clms.SearchPage(
            rows=[
                {
                    "date": "2026-11-23",
                    "author_name": "山田太郎",
                    "author_mail": "",
                    "title": "コード例",
                    "tags": "",
                    "body": (
                        "説明 https://example.com\n"
                        "```python\n"
                        "print('<script>')\n"
                        "https://example.org\n"
                        "```\n"
                        "続きの説明"
                    ),
                }
            ],
            total=1,
            page=1,
            limit=10,
        )

        rendered = clms.render_search_page("", result)

        self.assertIn('<div class="code-language">python</div>', rendered)
        self.assertIn("print(&#x27;&lt;script&gt;&#x27;)", rendered)
        self.assertIn('href="https://example.com"', rendered)
        self.assertNotIn('href="https://example.org"', rendered)
        self.assertIn('<div class="entry-body-text">続きの説明</div>', rendered)

    def test_unclosed_code_fence_continues_to_end_of_body(self):
        rendered = clms.render_entry_body("```text\n<unsafe>\n")

        self.assertIn('<div class="code-language">text</div>', rendered)
        self.assertIn("&lt;unsafe&gt;", rendered)
        self.assertNotIn("entry-body-text", rendered)


class HttpTests(unittest.TestCase):
    def test_search_and_css_endpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            dbfile = Path(directory) / "memo.sqlite3"
            cssfile = Path(directory) / "clms.css"
            cssfile.write_text("body { color: red; }", encoding="utf-8")
            clms.initialize_database(dbfile)
            clms.sync_database(dbfile, clms.parse_source(SAMPLE))
            embedder = FakeEmbedder()
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                clms.make_handler(dbfile, cssfile, embedder=embedder),
            )
            self.assertEqual(server.server_address[0], clms.LOOPBACK_HOST)
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                base_url = f"http://127.0.0.1:{server.server_port}"
                with urlopen(f"{base_url}/search?q=SQLite&page=1&limit=1") as response:
                    page = response.read().decode("utf-8")
                    self.assertEqual(response.status, 200)
                    self.assertIn("SQLiteの検索機能", page)
                    self.assertIn("次へ", page)
                    self.assertIn('<script src="/shortcuts.js" defer></script>', page)
                    self.assertIn('rel="next"', page)
                    self.assertIn('option value="hybrid" selected', page)
                    self.assertIn("mode=hybrid", page)
                with urlopen(f"{base_url}/search?q=SQLite&page=2&limit=1") as response:
                    page = response.read().decode("utf-8")
                    self.assertIn('rel="prev"', page)
                with urlopen(f"{base_url}/shortcuts.js") as response:
                    script = response.read().decode("utf-8")
                    self.assertEqual(
                        response.headers.get_content_type(), "application/javascript"
                    )
                    self.assertIn('event.code === "KeyJ"', script)
                    self.assertIn('event.code === "Comma"', script)
                with urlopen(f"{base_url}/search?q=SQLite&mode=bm25") as response:
                    page = response.read().decode("utf-8")
                    self.assertIn('option value="bm25" selected', page)
                with self.assertRaises(HTTPError) as error:
                    urlopen(f"{base_url}/search?q=SQLite&mode=unknown")
                self.assertEqual(error.exception.code, 400)
                with urlopen(f"{base_url}/style.css") as response:
                    self.assertEqual(response.read(), b"body { color: red; }")
            finally:
                server.shutdown()
                server.server_close()
                worker.join(timeout=2)


class NetworkAccessTests(unittest.TestCase):
    def test_only_loopback_addresses_are_allowed(self):
        self.assertTrue(clms._is_loopback_address("127.0.0.1"))
        self.assertTrue(clms._is_loopback_address("::1"))
        self.assertFalse(clms._is_loopback_address("192.168.1.10"))
        self.assertFalse(clms._is_loopback_address("not-an-ip-address"))


class MainTests(unittest.TestCase):
    def test_bind_error_is_reported_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "memo.txt"
            dbfile = Path(directory) / "memo.sqlite3"
            source.write_text(SAMPLE, encoding="utf-8")
            with patch.object(
                clms, "ThreadingHTTPServer", side_effect=PermissionError("port denied")
            ) as server_constructor:
                with self.assertLogs("clms", level="ERROR") as captured:
                    result = clms.main(
                        [
                            "--source", str(source),
                            "--dbfile", str(dbfile),
                            "--port", "18080",
                            "--search-mode", "bm25",
                        ]
                    )
            self.assertEqual(result, 1)
            self.assertIn("待ち受けできません", captured.output[0])
            self.assertIn("port denied", captured.output[0])
            self.assertEqual(
                server_constructor.call_args.args[0], (clms.LOOPBACK_HOST, 18080)
            )


if __name__ == "__main__":
    unittest.main()
