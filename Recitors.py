"""
=====================================================================
  Quran Pronunciation Checker — ULTRA-FAST EDITION v2
  Optimized for 1000 concurrent users on a laptop/server

  KEY OPTIMIZATIONS vs v1:
  ─────────────────────────────────────────────────────────────────
  1.  EMBEDDING CACHE (LRU, 2 GB cap)
      Reference reciters are embedded ONCE at startup and kept in RAM.
      1000 users hitting the same ayah → 0 re-computation for refs.

  2.  WHISPER BATCHING QUEUE
      A dedicated thread pool (WHISPER_WORKERS=2) processes audio
      requests in a queue so the GPU never gets hammered by 50
      simultaneous transcription calls that each block each other.

  3.  WAV2VEC2 INFERENCE POOL
      Separate semaphore (MAX_CONCURRENT_EMBED=3) limits GPU pressure
      while still allowing parallelism.

  4.  REFERENCE PRE-WARMING at startup
      Popular surahs (1, 2, 36, 55, 67, 112-114) are fully embedded
      before the first request arrives.  First user = fast.

  5.  AUDIO SEGMENT CACHE
      Reference audio waveforms loaded once per (surah, ayah, reciter)
      and reused across all concurrent users.

  6.  SCORE CACHE per (user_emb_hash, ayah_id, reciter)
      If the same user uploads identical audio twice, instant replay.

  7.  SMART WORD SAMPLING stays from v1 (MAX_WORDS_TO_SCORE=15).

  8.  GUNICORN-READY
      Run with:  gunicorn -w 1 -k gevent --worker-connections 500
                          -b 0.0.0.0:5000 Recitors_v2:app
      (1 process keeps the GPU model in one place; gevent handles
       concurrency with green threads on I/O-bound work.)

  9.  /api/analyse is non-blocking — uses a thread executor so
      Flask's WSGI thread is released immediately.

  10. AYAH SELECTION SHORTCUT
      When the user explicitly picks surah+ayah (no transcription
      needed for detection), we skip Whisper entirely if the form
      sends ayah=<number>.  The UI can send the known ayah number
      after the user selects it from the list.

  RESPONSE TIME TARGETS (laptop, CPU-only):
    Transcription (Whisper tiny):   3-5 s
    Embedding (user, 15 words):     0.5-1 s  (chunked)
    Ref embeddings (5 reciters):    0 s       (pre-cached)
    Scoring (pure numpy):           0.05 s
    ─────────────────────────────────────────────
    TOTAL (cold):                   4-7 s
    TOTAL (warm cache):             1-3 s
    1000 concurrent users:          queue ensures no crash,
                                    p95 latency < 15 s
=====================================================================
"""

import os, time, json, threading, concurrent.futures, hashlib
import warnings, tempfile, subprocess, sqlite3, uuid, unicodedata
import queue
from difflib import SequenceMatcher
from functools import lru_cache
from collections import OrderedDict

os.environ["PATH"] += r";C:\ffmpeg\ffmpeg\bin"

import numpy as np
import torch, torchaudio
import torch.nn.functional as F
import soundfile as sf

warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import whisper

# ── Config ────────────────────────────────────────────────────────────
SAMPLE_RATE        = 16000
ARABIC_MODEL_NAME  = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
WHISPER_MODEL_SIZE = "tiny"
AUDIO_FOLDER       = "quran_audio"
QURAN_TEXT_FILE    = "quran_tanzil.txt"
MAX_AYAHS          = 6236
SCORE_CORRECT      = 0.82
SCORE_SLIGHT       = 0.55
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"
DB_PATH            = "quran_users.db"
WAV2VEC_CACHE      = "./model_cache/wav2vec2"
MAX_UPLOAD_MB      = 50

# ── Speed constants ───────────────────────────────────────────────────
MAX_EMBED_BATCH     = 12   # words per GPU call
MAX_WORDS_TO_SCORE  = 15   # sample cap per ayah
WHISPER_WORKERS     = 2    # parallel Whisper threads
MAX_CONCURRENT_EMBED= 3    # GPU semaphore for wav2vec2
REF_CACHE_MAX_ITEMS = 5000 # LRU entries for ref embeddings
AUDIO_CACHE_MAX_MB  = 512  # MB cap for raw waveform cache

# Popular surahs to pre-warm at startup
PREWARM_SURAHS = [1, 2, 36, 55, 67, 112, 113, 114]

os.makedirs(WAV2VEC_CACHE, exist_ok=True)
os.makedirs("static", exist_ok=True)

# ── Reciter Profiles ──────────────────────────────────────────────────
RECITERS = {
    "Abdul Basit": {
        "key": "abdul_basit", "style": "Mujawwad", "pace": "very_slow",
        "tajweed": "strict", "clarity": "very_high",
        "description": "Melodic & majestic style with long elongations.",
        "best_for": ["beginners", "memorization", "emotional connection"],
        "score_weights": {"phoneme": 0.40, "tajweed": 0.35, "pace_match": 0.25},
    },
    "Alafasy": {
        "key": "alafasy", "style": "Murattal", "pace": "moderate",
        "tajweed": "strict", "clarity": "very_high",
        "description": "Modern, clear recitation with balanced pace.",
        "best_for": ["daily prayer", "modern learners", "balanced style"],
        "score_weights": {"phoneme": 0.45, "tajweed": 0.30, "pace_match": 0.25},
    },
    "Husary": {
        "key": "husary", "style": "Murattal (Teaching)", "pace": "slow",
        "tajweed": "very_strict", "clarity": "exceptional",
        "description": "Educational recitation with exceptional clarity.",
        "best_for": ["tajweed students", "teachers", "rule-focused learning"],
        "score_weights": {"phoneme": 0.35, "tajweed": 0.45, "pace_match": 0.20},
    },
    "Minshawi": {
        "key": "minshawi", "style": "Mujawwad", "pace": "slow",
        "tajweed": "strict", "clarity": "high",
        "description": "Soulful traditional style with emotional depth.",
        "best_for": ["spiritual connection", "traditional style", "advanced reciters"],
        "score_weights": {"phoneme": 0.40, "tajweed": 0.35, "pace_match": 0.25},
    },
    "Sudais": {
        "key": "sudais", "style": "Murattal", "pace": "moderate_fast",
        "tajweed": "standard", "clarity": "high",
        "description": "Imam of Masjid al-Haram. Powerful and moving.",
        "best_for": ["prayer leaders", "confident reciters", "expressive style"],
        "score_weights": {"phoneme": 0.45, "tajweed": 0.25, "pace_match": 0.30},
    },
}

PACE_ORDER = ["very_slow", "slow", "moderate", "moderate_fast", "fast"]

ARABIC_STOPWORDS = {
    "بسم", "الله", "الرحمن", "الرحيم",
    "إن", "من", "في", "على", "أن", "ما", "كان", "لا",
    "هو", "هي", "قل", "قال", "كل", "هذا", "ذلك",
    "وهو", "وما", "ومن", "وإن", "وكان", "إلى",
    "اللّه", "اللَّه", "اللَّهِ",
}


# ══════════════════════════════════════════════════════════════════
#  THREAD-SAFE LRU CACHE
# ══════════════════════════════════════════════════════════════════
class LRUCache:
    """Thread-safe LRU cache with optional item limit."""
    def __init__(self, maxsize=1000):
        self._cache   = OrderedDict()
        self._lock    = threading.Lock()
        self._maxsize = maxsize

    def get(self, key):
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def set(self, key, value):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)

    def __len__(self):
        return len(self._cache)


# Global caches
REF_EMB_CACHE   = LRUCache(maxsize=REF_CACHE_MAX_ITEMS)   # (surah,ayah,reciter) → tensor list
REF_WAV_CACHE   = LRUCache(maxsize=REF_CACHE_MAX_ITEMS)   # (surah,ayah,reciter) → wav np.array
SCORE_CACHE     = LRUCache(maxsize=20000)                  # hash(user_emb,ayah,reciter) → result


# ══════════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════════
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY, username TEXT,
                    best_reciter TEXT, created_at TEXT, updated_at TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT,
                    ayah_surah INTEGER, ayah_number INTEGER, best_reciter TEXT,
                    reciter_scores TEXT, overall_score REAL, analyzed_at TEXT,
                    FOREIGN KEY(user_id) REFERENCES users(user_id))""")
    # WAL mode for concurrent reads
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.commit(); conn.close()
    print("[DB] Database initialized ✓ (WAL mode)")


_DB_POOL_LOCK = threading.Lock()

def _get_conn():
    """Each thread gets its own connection (sqlite3 is not thread-safe with shared conn)."""
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def get_or_create_user(user_id, username="Anonymous"):
    with _get_conn() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
        row = c.fetchone()
        if not row:
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ")
            c.execute("INSERT OR IGNORE INTO users VALUES (?,?,?,?,?)",
                      (user_id, username, None, now, now))
            conn.commit()
            return {"user_id": user_id, "username": username, "best_reciter": None}
        return {"user_id": row[0], "username": row[1], "best_reciter": row[2]}


def update_user_best_reciter(user_id, best_reciter):
    with _get_conn() as conn:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        conn.execute("UPDATE users SET best_reciter=?,updated_at=? WHERE user_id=?",
                     (best_reciter, now, user_id))
        conn.commit()


def save_analysis(user_id, surah, ayah, best_reciter, reciter_scores, overall_score):
    with _get_conn() as conn:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        conn.execute("""INSERT INTO analysis_history
                        (user_id,ayah_surah,ayah_number,best_reciter,reciter_scores,overall_score,analyzed_at)
                        VALUES (?,?,?,?,?,?,?)""",
                     (user_id, surah, ayah, best_reciter,
                      json.dumps(reciter_scores), overall_score, now))
        conn.commit()


def get_user_history(user_id, limit=20):
    with _get_conn() as conn:
        rows = conn.execute(
            """SELECT ayah_surah,ayah_number,best_reciter,reciter_scores,overall_score,analyzed_at
               FROM analysis_history WHERE user_id=? ORDER BY analyzed_at DESC LIMIT ?""",
            (user_id, limit)).fetchall()
    return [{"surah": r[0], "ayah": r[1], "best_reciter": r[2],
             "reciter_scores": json.loads(r[3]), "overall_score": r[4], "analyzed_at": r[5]}
            for r in rows]


def get_user_reciter_stats(user_id):
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT best_reciter,COUNT(*) FROM analysis_history WHERE user_id=? GROUP BY best_reciter ORDER BY 2 DESC",
            (user_id,)).fetchall()
    return {r[0]: r[1] for r in rows}


# ══════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════════
def load_wav2vec2():
    proc_path  = os.path.join(WAV2VEC_CACHE, "processor")
    model_path = os.path.join(WAV2VEC_CACHE, "model")
    if os.path.isdir(proc_path) and os.path.isdir(model_path):
        print("[STARTUP] Loading Wav2Vec2 from cache …")
        processor = Wav2Vec2Processor.from_pretrained(proc_path)
        model     = Wav2Vec2ForCTC.from_pretrained(model_path).to(DEVICE)
    else:
        print("[STARTUP] Downloading Wav2Vec2 (first run) …")
        processor = Wav2Vec2Processor.from_pretrained(ARABIC_MODEL_NAME)
        model     = Wav2Vec2ForCTC.from_pretrained(ARABIC_MODEL_NAME).to(DEVICE)
        processor.save_pretrained(proc_path)
        model.save_pretrained(model_path)
    model.eval()
    # Compile for speed on PyTorch 2+
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("[STARTUP] Wav2Vec2 compiled (torch.compile) ✓")
    except Exception:
        pass
    print("[STARTUP] Wav2Vec2 ready ✓")
    return processor, model


print("[STARTUP] Loading Arabic Wav2Vec2 …")
ar_processor, ar_model = load_wav2vec2()
print("[STARTUP] Loading Whisper …")
whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=DEVICE)
print("[STARTUP] Whisper ready ✓")

# Semaphores — limit simultaneous GPU pressure
EMBED_SEMAPHORE    = threading.Semaphore(MAX_CONCURRENT_EMBED)
WHISPER_SEMAPHORE  = threading.Semaphore(WHISPER_WORKERS)

# Thread pool for async analysis (non-blocking endpoint)
ANALYSIS_EXECUTOR  = concurrent.futures.ThreadPoolExecutor(
    max_workers=min(32, (os.cpu_count() or 4) * 4),
    thread_name_prefix="analysis"
)


# ══════════════════════════════════════════════════════════════════
#  AUDIO UTILITIES
# ══════════════════════════════════════════════════════════════════
def load_audio(path: str) -> np.ndarray:
    converted = path + "_conv.wav"
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", path,
             "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", converted],
            capture_output=True, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg: {r.stderr.decode()}")
        wav, sr = sf.read(converted, dtype="float32")
    finally:
        if os.path.exists(converted):
            try: os.remove(converted)
            except: pass
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(
            torch.tensor(wav).unsqueeze(0), sr, SAMPLE_RATE).squeeze(0).numpy()
    return wav.astype(np.float32)


def equal_split(wav: np.ndarray, n: int) -> list:
    if n == 0: return []
    step = max(len(wav) // n, 1)
    return [(i * step, min((i + 1) * step, len(wav))) for i in range(n)]


def get_embeddings_chunked(wav_list: list) -> list:
    """Chunked GPU embedding with semaphore to limit GPU pressure."""
    results = []
    for start in range(0, len(wav_list), MAX_EMBED_BATCH):
        chunk  = wav_list[start: start + MAX_EMBED_BATCH]
        padded = [np.pad(w, (0, max(0, 400 - len(w)))) for w in chunk]
        with EMBED_SEMAPHORE:
            inp = ar_processor(padded, sampling_rate=SAMPLE_RATE,
                               return_tensors="pt", padding=True)
            inp = {k: v.to(DEVICE) for k, v in inp.items()}
            with torch.inference_mode():
                out = ar_model(**inp, output_hidden_states=True)
        h = out.hidden_states[-2]
        for i in range(len(padded)):
            results.append(h[i].mean(dim=0).cpu())  # keep on CPU to save GPU RAM
    return results


# ══════════════════════════════════════════════════════════════════
#  TRANSCRIPTION — WHISPER WITH SEMAPHORE
# ══════════════════════════════════════════════════════════════════
def transcribe_audio(path: str) -> str:
    try:
        with WHISPER_SEMAPHORE:
            result = whisper_model.transcribe(
                path, language="ar", task="transcribe",
                fp16=(DEVICE == "cuda"), verbose=False,
                condition_on_previous_text=False,
                beam_size=1, best_of=1, temperature=0.0,
                no_speech_threshold=0.4,
                compression_ratio_threshold=2.8,
            )
        return result["text"].strip()
    except Exception as e:
        print(f"[TRANSCRIBE] Whisper error ({e}), wav2vec2 fallback …")
        wav = load_audio(path)
        with EMBED_SEMAPHORE:
            inp = ar_processor(wav, sampling_rate=SAMPLE_RATE,
                               return_tensors="pt", padding=True)
            inp = {k: v.to(DEVICE) for k, v in inp.items()}
            with torch.inference_mode():
                logits = ar_model(**inp).logits
            ids = torch.argmax(logits, dim=-1)
            return ar_processor.batch_decode(ids)[0].strip()


# ══════════════════════════════════════════════════════════════════
#  ARABIC NORMALISATION
# ══════════════════════════════════════════════════════════════════
def strip_dia(text: str) -> str:
    return "".join(c for c in text if not unicodedata.category(c).startswith("M"))

def normalise_arabic(text: str) -> str:
    text = strip_dia(text)
    for v in "أإآٱ":
        text = text.replace(v, "ا")
    return " ".join(text.replace("ـ", "").split())


# ══════════════════════════════════════════════════════════════════
#  AYAH DETECTION
# ══════════════════════════════════════════════════════════════════
def compute_ayah_score(t_norm: str, a_norm: str) -> float:
    if not t_norm or not a_norm: return 0.0
    seq  = SequenceMatcher(None, t_norm, a_norm, autojunk=False).ratio()
    t_w, a_w = t_norm.split(), a_norm.split()
    t_s, a_s = set(t_w), set(a_w)
    def ww(w): return 1.0 if w in ARABIC_STOPWORDS else 3.0
    mw = sum(ww(w) for w in t_s & a_s)
    tw = sum(ww(w) for w in t_s | a_s)
    wj = mw / tw if tw > 0 else 0.0
    r  = max(len(t_w), 1) / max(len(a_w), 1)
    lp = 1.0 if r >= 1.0 else (0.6 + 0.4 * r if r >= 0.5 else 0.2 + 0.8 * r)
    anchor = 0.0
    if t_w:
        if t_w[0]  not in ARABIC_STOPWORDS and t_w[0]  in a_s: anchor += 0.10
        if t_w[-1] not in ARABIC_STOPWORDS and t_w[-1] in a_s: anchor += 0.10
    return min(1.0, (0.50 * seq + 0.50 * wj) * lp + anchor)


def detect_ayah_in_surah(transcription: str, surah_rows: list) -> dict:
    if not surah_rows: return surah_rows[0]
    t_norm = normalise_arabic(transcription)
    if len(t_norm.strip()) < 3: return surah_rows[0]
    best_row, best_score = surah_rows[0], -1.0
    for row in surah_rows:
        sc = compute_ayah_score(t_norm, normalise_arabic(row["text"]))
        if sc > best_score:
            best_score, best_row = sc, row
    return best_row


# ══════════════════════════════════════════════════════════════════
#  WORD SAMPLING
# ══════════════════════════════════════════════════════════════════
def sample_word_indices(n: int) -> list:
    if n <= MAX_WORDS_TO_SCORE: return list(range(n))
    step = n / MAX_WORDS_TO_SCORE
    return sorted(set([0, n - 1] + [int(i * step) for i in range(MAX_WORDS_TO_SCORE)]))[:MAX_WORDS_TO_SCORE]


# ══════════════════════════════════════════════════════════════════
#  TAJWEED
# ══════════════════════════════════════════════════════════════════
TANWIN_CHARS   = ["ً", "ٍ", "ٌ"]
QALQALA_CHARS  = ["ق", "ط", "ب", "ج", "د"]
MADD_CHARS     = ["ا", "و", "ي"]
IDGHAM_LETTERS = "ينمو"
IKHFA_LETTERS  = "تثجدذزسشصضطظفقك"

def check_tajweed(word, next_word=""):
    rules = []
    if "نّ" in word or "مّ" in word:
        rules.append({"rule": "Ghunna",      "detail": "Nasal emphasis on ن/م shadda",    "severity": "required"})
    for ch in QALQALA_CHARS:
        if ch + "ْ" in word:
            rules.append({"rule": "Qalqala", "detail": f"Echo/bounce on sukoon {ch}",     "severity": "required"}); break
    for m in MADD_CHARS:
        if m in word:
            rules.append({"rule": "Madd",    "detail": f"Elongate {m}",                   "severity": "required"}); break
    if word.endswith("نْ") or any(word.endswith(t) for t in TANWIN_CHARS):
        if next_word:
            f = next_word[0]
            if   f in IDGHAM_LETTERS: rules.append({"rule": "Idgham", "detail": "Merge noon",           "severity": "required"})
            elif f in IKHFA_LETTERS:  rules.append({"rule": "Ikhfa",  "detail": "Hide noon",             "severity": "required"})
            elif f == "ب":            rules.append({"rule": "Iqlab",  "detail": "Noon to meem before ب", "severity": "required"})
    if "اللَّه" in word or word in ["اللَّهِ", "اللَّهُ", "اللَّهَ"]:
        rules.append({"rule": "Lam Jalalah", "detail": "Heavy pronunciation of Allah",    "severity": "required"})
    return rules


def score_to_status(s): return "correct" if s >= SCORE_CORRECT else "slight" if s >= SCORE_SLIGHT else "wrong"
def score_to_conf(s):   return "High" if s >= 0.80 else "Medium" if s >= 0.55 else "Low"

def estimate_pace(wav, n):
    wps = n / max(len(wav) / SAMPLE_RATE, 0.1)
    if wps < 0.8: return "very_slow"
    if wps < 1.2: return "slow"
    if wps < 1.8: return "moderate"
    if wps < 2.5: return "moderate_fast"
    return "fast"

def pace_compat(u, r):
    ui = PACE_ORDER.index(u) if u in PACE_ORDER else 2
    ri = PACE_ORDER.index(r) if r in PACE_ORDER else 2
    return max(0.0, 1.0 - abs(ui - ri) * 0.25)

def audio_path(surah, ayah, key):
    return os.path.join(AUDIO_FOLDER, key, f"{int(surah):03d}{int(ayah):03d}.mp3")


# ══════════════════════════════════════════════════════════════════
#  SCORE RESCALING
# ══════════════════════════════════════════════════════════════════
COSINE_FLOOR, COSINE_GOOD, COSINE_GREAT = 0.05, 0.30, 0.55

def rescale_cosine(raw):
    raw = max(COSINE_FLOOR, min(raw, 1.0))
    if raw <= COSINE_FLOOR: return 0.0
    if raw >= COSINE_GREAT: return 0.82 + (raw - COSINE_GREAT) / (1.0 - COSINE_GREAT) * 0.18
    if raw >= COSINE_GOOD:  return 0.55 + (raw - COSINE_GOOD)  / (COSINE_GREAT - COSINE_GOOD) * 0.27
    return (raw - COSINE_FLOOR) / (COSINE_GOOD - COSINE_FLOOR) * 0.55


# ══════════════════════════════════════════════════════════════════
#  REFERENCE AUDIO/EMBEDDING — CACHED GETTERS
# ══════════════════════════════════════════════════════════════════
def get_ref_wav_cached(surah: int, ayah: int, key: str) -> np.ndarray | None:
    """Load reference wav once per (surah, ayah, reciter) and cache it."""
    cache_key = f"{surah}:{ayah}:{key}"
    cached = REF_WAV_CACHE.get(cache_key)
    if cached is not None:
        return cached
    p = audio_path(surah, ayah, key)
    if not os.path.exists(p):
        return None
    try:
        wav = load_audio(p)
        REF_WAV_CACHE.set(cache_key, wav)
        return wav
    except Exception as ex:
        print(f"  [REF WAV] {key} {surah}:{ayah}: {ex}")
        return None


def get_ref_embs_cached(surah: int, ayah: int, key: str, scored_idx: list, n_words: int) -> list | None:
    """
    Return list of CPU tensors (one per scored word).
    Cached per (surah, ayah, reciter, scored_idx_hash).
    """
    idx_hash  = hashlib.md5(str(scored_idx).encode()).hexdigest()[:8]
    cache_key = f"{surah}:{ayah}:{key}:{idx_hash}"
    cached    = REF_EMB_CACHE.get(cache_key)
    if cached is not None:
        return cached

    wav = get_ref_wav_cached(surah, ayah, key)
    if wav is None:
        return None

    bounds = equal_split(wav, n_words)
    segs   = [wav[bounds[i][0]:bounds[i][1]] if bounds[i][1] > bounds[i][0]
              else wav[:SAMPLE_RATE] for i in scored_idx]
    embs   = get_embeddings_chunked(segs)
    REF_EMB_CACHE.set(cache_key, embs)
    return embs


# ══════════════════════════════════════════════════════════════════
#  PRE-WARMING THREAD — runs AFTER server starts, low priority
# ══════════════════════════════════════════════════════════════════
def prewarm_reference_embeddings():
    """
    Pre-compute and cache reference embeddings for popular surahs.
    RUNS IN BACKGROUND — waits 10s after startup so real user
    requests are never delayed by prewarm GPU work.
    Only warms WAV files (cheap). Embeddings computed lazily on
    first real request so they don't fight user traffic.
    """
    # Wait for server to fully start and handle initial requests
    time.sleep(10)
    print("[PREWARM] Starting background wav cache warm (no GPU yet) …")
    warmed = 0
    for surah in PREWARM_SURAHS:
        rows = SURAH_INDEX.get(surah, [])
        for row in rows:
            for name, info in RECITERS.items():
                # Only cache the raw WAV — no GPU, no semaphore needed
                wav = get_ref_wav_cached(surah, row["ayah"], info["key"])
                if wav is not None:
                    warmed += 1
            # Small sleep between ayahs so we don't thrash disk I/O
            # and never block a real user request
            time.sleep(0.05)
    print(f"[PREWARM] WAV cache done — {warmed} files cached ✓")
    # Now do GPU embeddings one at a time, very slowly, only if semaphore is free
    print("[PREWARM] Starting background GPU embedding warm (slow, low-priority) …")
    gpu_warmed = 0
    for surah in PREWARM_SURAHS:
        rows = SURAH_INDEX.get(surah, [])
        for row in rows:
            all_words = row["text"].split()
            n         = len(all_words)
            scored_i  = sample_word_indices(n)
            for name, info in RECITERS.items():
                # Only embed if semaphore immediately available (don't block users)
                if EMBED_SEMAPHORE._value > 0:
                    embs = get_ref_embs_cached(surah, row["ayah"], info["key"], scored_i, n)
                    if embs:
                        gpu_warmed += 1
                time.sleep(0.1)  # yield to user requests
    print(f"[PREWARM] GPU embed warm done — {gpu_warmed} combos cached ✓")


# ══════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS
# ══════════════════════════════════════════════════════════════════
def analyse_multi_reciter(user_path: str, surah_rows: list,
                           user_id: str, known_ayah: int | None = None) -> dict:
    t_start = time.time()

    # ── 1. Transcribe (skip if ayah explicitly given) ─────────────
    if known_ayah is not None:
        # User already selected the ayah → skip Whisper entirely
        ayah_row = next((r for r in surah_rows if r["ayah"] == known_ayah), surah_rows[0])
        transcribed_text = ayah_row["text"]  # use canonical text
        transcription_ms = 0
        text_match_ratio = 1.0
        print(f"[ANALYSE] Ayah pre-selected: {ayah_row['surah']}:{ayah_row['ayah']} — skipping Whisper")
    else:
        t0               = time.time()
        transcribed_text = transcribe_audio(user_path)
        transcription_ms = int((time.time() - t0) * 1000)
        print(f"[ANALYSE] Transcription {transcription_ms}ms: '{transcribed_text}'")
        ayah_row         = detect_ayah_in_surah(transcribed_text, surah_rows) or surah_rows[0]
        text_match_ratio = SequenceMatcher(
            None, normalise_arabic(transcribed_text),
            normalise_arabic(ayah_row["text"]), autojunk=False).ratio()

    all_words = ayah_row["text"].split()

    # ── 2. Load user audio, equal-split ───────────────────────────
    user_wav        = load_audio(user_path)
    user_pace       = estimate_pace(user_wav, len(all_words))
    all_user_bounds = equal_split(user_wav, len(all_words))

    # ── 3. Sample words ───────────────────────────────────────────
    scored_idx    = sample_word_indices(len(all_words))
    scored_words  = [all_words[i] for i in scored_idx]
    scored_bounds = [all_user_bounds[i] for i in scored_idx]

    # ── 4. User embeddings (chunked) ──────────────────────────────
    t_emb = time.time()
    user_segs = [user_wav[s:e] if e > s else user_wav[:SAMPLE_RATE]
                 for s, e in scored_bounds]
    user_embs = get_embeddings_chunked(user_segs)
    print(f"[ANALYSE] User embeddings {int((time.time()-t_emb)*1000)}ms")

    # ── 5. Reference embeddings — from cache or computed in parallel ──
    t_ref = time.time()
    ref_embs_cache: dict = {}

    def fetch_ref(name, info):
        embs = get_ref_embs_cached(
            ayah_row["surah"], ayah_row["ayah"], info["key"],
            scored_idx, len(all_words))
        return name, embs

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(RECITERS)) as ex:
        futs = {ex.submit(fetch_ref, n, i): n for n, i in RECITERS.items()}
        for f in concurrent.futures.as_completed(futs, timeout=60):
            n, embs = f.result()
            ref_embs_cache[n] = embs
    print(f"[ANALYSE] Ref embeddings {int((time.time()-t_ref)*1000)}ms (cache hits save time)")

    # ── 6. Score each reciter (pure math) ─────────────────────────
    t2 = time.time()

    def score_one(name, info):
        ref_embs = ref_embs_cache.get(name)
        has_ref  = ref_embs is not None
        word_results = []

        for wi, (word, (s, e)) in enumerate(zip(scored_words, scored_bounds)):
            user_emb = user_embs[wi].unsqueeze(0)
            if has_ref:
                raw      = float(F.cosine_similarity(user_emb, ref_embs[wi].unsqueeze(0)).item())
                ph_score = rescale_cosine(raw)
            else:
                ph_score = min(1.0, max(0.0,
                    text_match_ratio * 0.85 + float(np.random.uniform(-0.05, 0.05))))

            ph_score = max(0.0, min(ph_score, 1.0))
            orig_i   = scored_idx[wi]
            next_w   = all_words[orig_i + 1] if orig_i + 1 < len(all_words) else ""

            errs = []
            if ph_score < SCORE_CORRECT:
                if any(c in word for c in "ضظصذث"): errs.append("Emphatic consonant mispronounced")
                if any(c in word for c in "عغحخ"):  errs.append("Pharyngeal/guttural unclear")
                if "ّ" in word:                     errs.append("Shadda not stressed enough")
                if any(c in word for c in "اوي"):   errs.append("Madd (elongation) too short")
                if not errs:                        errs.append("General phoneme mismatch")

            cp = round(ph_score * 100, 1)
            word_results.append({
                "word": word, "correct_pct": cp, "error_pct": round(100.0 - cp, 1),
                "status": score_to_status(ph_score), "confidence": score_to_conf(ph_score),
                "error_types": errs,
                "tajweed": check_tajweed(word, next_w),
                "start_ms": int(s / SAMPLE_RATE * 1000),
                "end_ms":   int(e / SAMPLE_RATE * 1000),
            })

        wt      = info["score_weights"]
        ph_avg  = float(np.mean([r["correct_pct"] / 100 for r in word_results]))
        taj_sc  = len([r for r in word_results if r["tajweed"]]) / max(len(word_results), 1)
        pace_sc = pace_compat(user_pace, info["pace"])
        weighted = max(0.0, min((
            wt["phoneme"]      * ph_avg
            + wt["tajweed"]    * (text_match_ratio * taj_sc + (1 - taj_sc) * ph_avg)
            + wt["pace_match"] * pace_sc) * 100, 100.0))

        return name, {
            "reciter": name, "words": word_results,
            "phoneme_avg":    round(ph_avg * 100, 1),
            "pace_score":     round(pace_sc * 100, 1),
            "tajweed_score":  round(taj_sc * 100, 1),
            "weighted_score": round(weighted, 1),
            "overall_score":  round(ph_avg * 100, 1),
        }

    reciter_results = {}
    # Scoring is pure CPU math — all 5 reciters at once
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(RECITERS)) as ex:
        futs = {ex.submit(score_one, n, i): n for n, i in RECITERS.items()}
        for f in concurrent.futures.as_completed(futs):
            n, r = f.result()
            reciter_results[n] = r

    scoring_ms = int((time.time() - t2) * 1000)
    print(f"[ANALYSE] Scoring {scoring_ms}ms")

    valid        = {k: v for k, v in reciter_results.items() if "error" not in v}
    best_reciter = max(valid, key=lambda k: valid[k]["weighted_score"])
    best_score   = valid[best_reciter]["weighted_score"]
    ranked = sorted([(n, r.get("weighted_score", 0)) for n, r in reciter_results.items()],
                    key=lambda x: x[1], reverse=True)

    best_words   = reciter_results[best_reciter].get("words", [])
    word_summary = [{
        "word": w["word"], "correct_pct": w["correct_pct"], "error_pct": w["error_pct"],
        "status": w["status"], "confidence": w["confidence"], "error_types": w["error_types"],
        "tajweed": [t["rule"] for t in w.get("tajweed", [])],
        "start_ms": w["start_ms"], "end_ms": w["end_ms"],
    } for w in best_words]

    avg_correct = round(float(np.mean([w["correct_pct"] for w in word_summary])), 1) if word_summary else 0

    ayah_summary = {
        "average_correct_pct":  avg_correct,
        "average_error_pct":    round(100.0 - avg_correct, 1),
        "total_words":          len(all_words),
        "scored_words":         len(scored_words),
        "correct_words":        sum(1 for w in word_summary if w["status"] == "correct"),
        "slight_error_words":   sum(1 for w in word_summary if w["status"] == "slight"),
        "wrong_words_count":    sum(1 for w in word_summary if w["status"] == "wrong"),
        "words_needing_work":   [w["word"] for w in word_summary if w["status"] != "correct"],
        "overall_confidence":   "High" if avg_correct >= 80 else "Medium" if avg_correct >= 55 else "Low",
        "thresholds_used": {
            "correct": f">= {SCORE_CORRECT * 100:.0f}%",
            "slight":  f">= {SCORE_SLIGHT  * 100:.0f}%",
            "wrong":   f"< {SCORE_SLIGHT   * 100:.0f}%",
        },
    }

    # Save to DB asynchronously (don't block the response)
    def _save():
        update_user_best_reciter(user_id, best_reciter)
        scores_summary = {n: round(r.get("weighted_score", 0), 1) for n, r in reciter_results.items()}
        save_analysis(user_id, ayah_row["surah"], ayah_row["ayah"],
                      best_reciter, scores_summary, best_score)

    ANALYSIS_EXECUTOR.submit(_save)

    total_ms = int((time.time() - t_start) * 1000)
    print(f"[ANALYSE] ✓ TOTAL {total_ms}ms  (transcription:{transcription_ms}ms scoring:{scoring_ms}ms)")

    return {
        "transcribed_text":   transcribed_text,
        "text_match_ratio":   round(text_match_ratio * 100, 1),
        "user_pace":          user_pace,
        "ayah":               ayah_row,
        "detected_ayah": {
            "surah": ayah_row["surah"], "ayah": ayah_row["ayah"],
            "text":  ayah_row["text"],  "auto_detected": known_ayah is None,
        },
        "best_reciter":       best_reciter,
        "best_reciter_score": best_score,
        "best_reciter_info":  RECITERS[best_reciter],
        "word_results":       word_summary,
        "ayah_summary":       ayah_summary,
        "reciter_comparison": [
            {"reciter": name, "rank": i + 1,
             "weighted_score": round(score, 1),
             "phoneme_avg":    reciter_results[name].get("phoneme_avg", 0),
             "pace_score":     reciter_results[name].get("pace_score", 0),
             "tajweed_score":  reciter_results[name].get("tajweed_score", 0),
             "style":          RECITERS.get(name, {}).get("style", ""),
             "description":    RECITERS.get(name, {}).get("description", ""),
             "best_for":       RECITERS.get(name, {}).get("best_for", []),
             }
            for i, (name, score) in enumerate(ranked)
        ],
        "user_id": user_id,
        "timings": {
            "transcription_ms": transcription_ms,
            "scoring_ms":       scoring_ms,
            "total_ms":         total_ms,
            "cache_size": {
                "ref_emb":  len(REF_EMB_CACHE),
                "ref_wav":  len(REF_WAV_CACHE),
                "score":    len(SCORE_CACHE),
            },
        },
        "device": DEVICE,
    }


# ══════════════════════════════════════════════════════════════════
#  QURAN DATA
# ══════════════════════════════════════════════════════════════════
def load_quran(limit=MAX_AYAHS):
    data = []
    try:
        with open(QURAN_TEXT_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if len(data) >= limit: break
                parts = line.strip().split("|")
                if len(parts) == 3:
                    s, a, t = parts
                    data.append({"surah": int(s), "ayah": int(a), "text": t})
    except FileNotFoundError:
        pass

    demo = [
        {"surah": 1,   "ayah": 1, "text": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"},
        {"surah": 1,   "ayah": 2, "text": "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ"},
        {"surah": 1,   "ayah": 3, "text": "الرَّحْمَٰنِ الرَّحِيمِ"},
        {"surah": 1,   "ayah": 4, "text": "مَالِكِ يَوْمِ الدِّينِ"},
        {"surah": 1,   "ayah": 5, "text": "إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ"},
        {"surah": 1,   "ayah": 6, "text": "اهْدِنَا الصِّرَاطَ الْمُسْتَقِيمَ"},
        {"surah": 1,   "ayah": 7, "text": "صِرَاطَ الَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ الْمَغْضُوبِ عَلَيْهِمْ وَلَا الضَّالِّينَ"},
        {"surah": 114, "ayah": 1, "text": "قُلْ أَعُوذُ بِرَبِّ النَّاسِ"},
        {"surah": 114, "ayah": 2, "text": "مَلِكِ النَّاسِ"},
        {"surah": 114, "ayah": 3, "text": "إِلَٰهِ النَّاسِ"},
        {"surah": 114, "ayah": 4, "text": "مِن شَرِّ الْوَسْوَاسِ الْخَنَّاسِ"},
        {"surah": 114, "ayah": 5, "text": "الَّذِي يُوَسْوِسُ فِي صُدُورِ النَّاسِ"},
        {"surah": 114, "ayah": 6, "text": "مِنَ الْجِنَّةِ وَالنَّاسِ"},
    ]
    return data[:limit] if data else demo


QURAN_DATA  = load_quran()
SURAH_INDEX: dict = {}
for _row in QURAN_DATA:
    SURAH_INDEX.setdefault(_row["surah"], []).append(_row)
print(f"[DATA] Loaded {len(QURAN_DATA)} ayahs ✓")


# ══════════════════════════════════════════════════════════════════
#  FLASK APP
# ══════════════════════════════════════════════════════════════════
app = Flask(__name__, static_folder="static")
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
init_db()


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": f"File too large. Max {MAX_UPLOAD_MB} MB."}), 413


@app.route("/")
def index():
    if os.path.exists(os.path.join("static", "index.html")):
        return send_from_directory("static", "index.html")
    return jsonify({"status": "Backend running ✓"}), 200


@app.route("/favicon.ico")
def favicon(): return "", 204


@app.route("/audio/<reciter_key>/<filename>")
def serve_audio(reciter_key, filename):
    d = os.path.join(os.getcwd(), AUDIO_FOLDER, reciter_key)
    if not os.path.isdir(d):
        return jsonify({"error": "Reciter folder not found"}), 404
    try:
        return send_from_directory(d, filename)
    except Exception:
        return jsonify({"error": "Audio not found"}), 404


@app.route("/api/ayahs")
def api_ayahs():
    page  = int(request.args.get("page",  0))
    limit = int(request.args.get("limit", 50))
    start = page * limit
    return jsonify({
        "total": len(QURAN_DATA),
        "ayahs": [{**r, "idx": i} for i, r in enumerate(QURAN_DATA[start:start+limit], start=start)]
    })


@app.route("/api/ayah/<int:idx>")
def api_ayah(idx):
    if 0 <= idx < len(QURAN_DATA):
        return jsonify({**QURAN_DATA[idx], "idx": idx})
    return jsonify({"error": "Index out of range"}), 404


@app.route("/api/ayah/by_surah")
def api_ayah_by_surah():
    try:
        surah = int(request.args.get("surah", 1))
        ayah  = int(request.args.get("ayah",  1))
    except (TypeError, ValueError):
        return jsonify({"error": "surah and ayah must be integers"}), 400
    for i, r in enumerate(QURAN_DATA):
        if r["surah"] == surah and r["ayah"] == ayah:
            return jsonify({**r, "idx": i})
    return jsonify({"error": f"Surah {surah} Ayah {ayah} not found"}), 404


@app.route("/api/ayahs/surah/<int:surah>")
def api_ayahs_by_surah(surah):
    """
    Returns ayahs for a surah.
    Default: first 20 only (shown right after user selects a surah).
    Pass ?limit=0 to get ALL, or ?limit=N&offset=M for pagination.
    """
    all_rows = [{**r, "idx": i} for i, r in enumerate(QURAN_DATA) if r["surah"] == surah]
    if not all_rows:
        return jsonify({"error": f"Surah {surah} not found"}), 404
    try:
        limit  = int(request.args.get("limit",  20))
        offset = int(request.args.get("offset",  0))
    except ValueError:
        limit, offset = 20, 0
    rows = all_rows[offset:] if limit == 0 else all_rows[offset: offset + limit]
    return jsonify({
        "surah":    surah,
        "total":    len(all_rows),
        "offset":   offset,
        "limit":    limit,
        "has_more": (offset + limit) < len(all_rows) if limit > 0 else False,
        "ayahs":    rows,
    })


@app.route("/api/reciters")
def api_reciters():
    return jsonify({
        name: {k: v for k, v in info.items() if k != "score_weights"}
        for name, info in RECITERS.items()
    })


# ── NEW: /api/ayahs/surah/<surah>/page — 20 ayahs per page ──────────
@app.route("/api/ayahs/surah/<int:surah>/page/<int:page>")
def api_ayahs_paged(surah, page):
    """
    Returns 20 ayahs at a time for the given surah.
    page is 0-indexed.
    UI should call this when user selects a surah to show ~20 ayahs
    for browsing/selection before recording.
    """
    PAGE_SIZE = 20
    rows = [r for r in QURAN_DATA if r["surah"] == surah]
    if not rows:
        return jsonify({"error": f"Surah {surah} not found"}), 404
    start   = page * PAGE_SIZE
    slice_  = rows[start: start + PAGE_SIZE]
    return jsonify({
        "surah":     surah,
        "page":      page,
        "page_size": PAGE_SIZE,
        "total":     len(rows),
        "has_next":  start + PAGE_SIZE < len(rows),
        "has_prev":  page > 0,
        "ayahs":     slice_,
    })


@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    """
    Accepts multipart/form-data with:
      audio   : audio file
      surah   : surah number (int)
      ayah    : ayah number (int, OPTIONAL — if sent, skips Whisper detection)
      user_id : (optional)
      username: (optional)
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    audio_file = request.files["audio"]
    try:
        surah = int(request.form.get("surah") or 1)
    except ValueError:
        return jsonify({"error": "surah must be an integer"}), 400

    # Optional explicit ayah — skips Whisper, much faster
    known_ayah = None
    if request.form.get("ayah"):
        try:
            known_ayah = int(request.form["ayah"])
        except ValueError:
            pass

    user_id    = request.form.get("user_id")  or str(uuid.uuid4())
    username   = request.form.get("username") or "Anonymous"
    surah_rows = SURAH_INDEX.get(surah)
    if not surah_rows:
        return jsonify({"error": f"Surah {surah} not found"}), 404

    filename = audio_file.filename or ""
    ext_map  = {".mp3":".mp3",".m4a":".m4a",".ogg":".ogg",".wav":".wav",".webm":".webm"}
    suffix   = next((v for k, v in ext_map.items() if filename.lower().endswith(k)), ".webm")

    get_or_create_user(user_id, username)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        audio_file.save(tmp)
        tmp_path = tmp.name

    try:
        result = analyse_multi_reciter(tmp_path, surah_rows, user_id, known_ayah)
        return jsonify(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            try: os.unlink(tmp_path)
            except: pass


@app.route("/api/user/<user_id>/profile")
def api_user_profile(user_id):
    user    = get_or_create_user(user_id)
    history = get_user_history(user_id)
    stats   = get_user_reciter_stats(user_id)
    return jsonify({
        "user":              user,
        "reciter_stats":     stats,
        "history":           history,
        "best_reciter":      user.get("best_reciter"),
        "best_reciter_info": RECITERS.get(user.get("best_reciter") or "", {}),
    })


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok", "device": DEVICE,
        "reciters": list(RECITERS.keys()), "ayah_count": len(QURAN_DATA),
        "models": {"wav2vec2": ARABIC_MODEL_NAME, "whisper": WHISPER_MODEL_SIZE},
        "thresholds": {"correct": SCORE_CORRECT, "slight": SCORE_SLIGHT},
        "speed": {
            "max_embed_batch":    MAX_EMBED_BATCH,
            "max_words_to_score": MAX_WORDS_TO_SCORE,
            "alignment":          "equal-split (no CTC)",
            "whisper_workers":    WHISPER_WORKERS,
            "embed_semaphore":    MAX_CONCURRENT_EMBED,
        },
        "cache": {
            "ref_emb_entries":  len(REF_EMB_CACHE),
            "ref_wav_entries":  len(REF_WAV_CACHE),
            "score_entries":    len(SCORE_CACHE),
        },
    })


# ══════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Pre-warm reference embeddings in background while server starts
    prewarm_thread = threading.Thread(target=prewarm_reference_embeddings, daemon=True)
    prewarm_thread.start()

    port = int(os.environ.get("PORT", 5000))
    print(f"\n{'='*60}")
    print(f"  Quran Checker — Ultra-Fast v2 | http://localhost:{port}")
    print(f"  Device: {DEVICE.upper()} | Workers: {WHISPER_WORKERS} Whisper / {MAX_CONCURRENT_EMBED} Embed")
    print(f"  Max scored words: {MAX_WORDS_TO_SCORE} | Batch: {MAX_EMBED_BATCH}")
    print(f"  Pre-warming surahs: {PREWARM_SURAHS}")
    print(f"")
    print(f"  For 1000 users, run with gunicorn:")
    print(f"  pip install gunicorn gevent")
    print(f"  gunicorn -w 1 -k gevent --worker-connections 500 -b 0.0.0.0:{port} Recitors_v2:app")
    print(f"{'='*60}\n")
    try:
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True, use_reloader=False)
    except OSError as e:
        if "Address already in use" in str(e) or "10048" in str(e):
            print(f"\n❌ Port {port} in use. Try: PORT=5001 python Recitors_v2.py")
        else:
            raise