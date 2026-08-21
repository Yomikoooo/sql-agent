import hashlib
import json
import os

from .tools import DB_PATH
from . import paths


ENABLED = os.getenv("LLM_CACHE", "0") == "1"
CACHE_FILE = paths.CACHE_DIR / "llm_cache.json"



def _data_version() -> str:    
    st = DB_PATH.stat()
    return f"{st.st_mtime_ns}:{st.st_size}"


class ExactCache:
    def __init__(self) -> None:
        self._data: dict[str, dict] = {}
        self.hits = 0
        self.misses = 0

    def load(self) -> None:
        if CACHE_FILE.exists():
            self._data = json.loads(CACHE_FILE.read_text())


    def save(self) -> None:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(self._data, ensure_ascii=False))


    def key(self, body: dict) -> str | None:
        if not ENABLED:
            return None
        if body.get("temperature", 0) > 0:
            return None
        raw = json.dumps(body, sort_keys=True, ensure_ascii=False)

        raw += "|" + _data_version()

        return hashlib.sha256(raw.encode()).hexdigest()
         

    def get(self, body: dict) -> dict | None:
        k = self.key(body)        
        if k is None or k not in self._data:
            self.misses += 1
            return None
        self.hits +=1
        return self._data[k]


    def put(self, body: dict, response: dict) -> None:
        k = self.key(body)
        if k is not None:
            self._data[k] = response


    def stats(self) -> str:
        total = self.hits + self.misses
        rate = self.hits / total * 100 if total else 0
        return f"cache {self.hits}/{total} = {rate:.1f}%"


CACHE = ExactCache()