"""Test-time dependency shims for optional/heavy third-party packages.

These keep unit tests fast and deterministic in minimal CI containers.
"""

from __future__ import annotations

import sys
import types
import os

os.environ.setdefault("PINECONE_API_KEY", "test-key-12345")


def _install_stub_modules() -> None:
    # pinecone shim
    if "pinecone" not in sys.modules:
        pinecone_mod = types.ModuleType("pinecone")

        class _FakePinecone:
            def __init__(self, *args, **kwargs):
                pass

            def list_indexes(self):
                return []

            def create_index(self, *args, **kwargs):
                return None

            class _Index:
                def upsert(self, *args, **kwargs):
                    return None

                def query(self, *args, **kwargs):
                    return {"matches": []}

                def delete(self, *args, **kwargs):
                    return None

                def describe_index_stats(self):
                    return {"total_vector_count": 0, "namespaces": {}}

            def Index(self, *args, **kwargs):
                return self._Index()

        class _FakeServerlessSpec:
            def __init__(self, *args, **kwargs):
                pass

        pinecone_mod.Pinecone = _FakePinecone
        pinecone_mod.ServerlessSpec = _FakeServerlessSpec
        sys.modules["pinecone"] = pinecone_mod

    # sentence-transformers shim
    if "sentence_transformers" not in sys.modules:
        st_mod = types.ModuleType("sentence_transformers")

        class _FakeEmbeddings(list):
            def tolist(self):
                return list(self)

        class _FakeSentenceTransformer:
            def __init__(self, *args, **kwargs):
                pass

            def encode(self, items, **kwargs):
                if not isinstance(items, list):
                    items = [items]
                return _FakeEmbeddings([[0.1] * 512 for _ in items])

        st_mod.SentenceTransformer = _FakeSentenceTransformer
        sys.modules["sentence_transformers"] = st_mod

    # whisper shim
    if "whisper" not in sys.modules:
        whisper_mod = types.ModuleType("whisper")

        class _FakeWhisperModel:
            def transcribe(self, *args, **kwargs):
                return {"text": "stub", "language": "en", "duration": 1.0, "segments": []}

        def _load_model(*args, **kwargs):
            return _FakeWhisperModel()

        whisper_mod.load_model = _load_model
        sys.modules["whisper"] = whisper_mod

    # PIL shim
    if "PIL" not in sys.modules:
        pil_mod = types.ModuleType("PIL")

        class _FakeImageModule:
            @staticmethod
            def open(*args, **kwargs):
                return object()

        pil_mod.Image = _FakeImageModule
        sys.modules["PIL"] = pil_mod

    # duckduckgo_search shim
    if "duckduckgo_search" not in sys.modules:
        ddg_mod = types.ModuleType("duckduckgo_search")

        class _FakeDDGS:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def text(self, *args, **kwargs):
                return []

        ddg_mod.DDGS = _FakeDDGS
        sys.modules["duckduckgo_search"] = ddg_mod


_install_stub_modules()

# pypdf shim
if "pypdf" not in sys.modules:
    pypdf_mod = types.ModuleType("pypdf")

    class _FakePdfReader:
        def __init__(self, *args, **kwargs):
            self.pages = []

    pypdf_mod.PdfReader = _FakePdfReader
    sys.modules["pypdf"] = pypdf_mod

# docx shim
if "docx" not in sys.modules:
    docx_mod = types.ModuleType("docx")

    class _FakeParagraph:
        text = ""

    class _FakeDocument:
        def __init__(self, *args, **kwargs):
            self.paragraphs = [_FakeParagraph()]

    docx_mod.Document = _FakeDocument
    sys.modules["docx"] = docx_mod

# requests shim
if "requests" not in sys.modules:
    requests_mod = types.ModuleType("requests")

    class _FakeResponse:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "stub", "models": []}

    def _fake_post(*args, **kwargs):
        return _FakeResponse()

    def _fake_get(*args, **kwargs):
        return _FakeResponse()

    requests_mod.post = _fake_post
    requests_mod.get = _fake_get
    requests_mod.exceptions = types.SimpleNamespace(RequestException=Exception)
    sys.modules["requests"] = requests_mod

# PyPDF2 shim (coordinator fallback extractor)
if "PyPDF2" not in sys.modules:
    pypdf2_mod = types.ModuleType("PyPDF2")

    class _FakePdfPage:
        @staticmethod
        def extract_text():
            return ""

    class _FakePdfReader:
        def __init__(self, *args, **kwargs):
            self.pages = [_FakePdfPage()]

    pypdf2_mod.PdfReader = _FakePdfReader
    sys.modules["PyPDF2"] = pypdf2_mod
