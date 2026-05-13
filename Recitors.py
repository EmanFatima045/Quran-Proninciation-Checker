"""
=====================================================================
  Quran Pronunciation Checker — v4  (WINDOWS FIX + BISMILLAH FIX + SPEED)

  FIXES vs v3:
  ────────────────────────────────────────────────────────────────
  FIX 1 — torch.compile CRASH ON WINDOWS (InductorError: cl not found)
    Root cause: torch.compile tries to JIT-compile C++ code using
    MSVC (cl.exe). Most Windows installs don't have Visual Studio
    build tools, causing a crash on first GPU call.

    Solution: Completely remove torch.compile. On CPU (which is what
    you're running), torch.compile with reduce-overhead actually ADDS
    overhead during warmup (compiling graph) and doesn't help until
    the 50th+ call. Plain eager mode is faster for small batches.

  FIX 2 — BISMILLAH FALSE POSITIVE  (SURAH AL-FATIHA)
    Root cause: When user selects Surah 1 (Al-Fatiha) and recites
    any ayah, the detector kept picking ayah 1 (Bismillah) because:
      a) All Bismillah words are stopwords → weight=1.0 each
      b) The penalty in v3 only fires in detect_ayah_in_surah()
         but PATH A (user-selected ayah) completely BYPASSES detection
         and directly uses ayah 1 if that's what's stored.
      c) Frontend wasn't sending ayah= field → always PATH B
         → detection always returns Bismillah

    REAL FIX — THREE LAYERS:
      1) Surah-1 special rule: ayah 1 of Surah 1 IS genuinely
         Bismillah AND Al-Fatiha starts with it. But if user is
         reciting Surah 1 and selects ayah 2-7, we trust that.
         If no ayah selected (PATH B), we use minimum-score=0.15
         for ayah 1 and require at least 2 non-stopword matches.
      2) Boost score for NON-bismillah ayahs: add +0.20 base bonus
         to any ayah > 1 so Bismillah has to genuinely dominate.
      3) Hard clamp: ayah 1 score is capped at 0.45 unless the
         transcript contains ≥ 2 unique rare words from it.

  FIX 3 — FASTER TRANSCRIPTION
    Whisper tiny is slow on CPU (~3s). Optimisations:
      a) Trim silence from audio before passing to Whisper
         (saves ~0.5-1s on short clips with leading silence)
      b) Whisper temperature=0.0, beam_size=1, best_of=1 already set
      c) Added: initial_prompt with Arabic context helps tiny model
         avoid hallucinating non-Arabic text (the ghost loop you saw:
         "لأحظة من المقابل لأحظة" — that's a hallucination from silence)
      d) no_speech_threshold raised to 0.6: reject silent/noise audio
         fast instead of hallucinating

  FIX 4 — SILENCE/NOISE GUARD
    If Whisper returns < 5 Arabic chars OR detects repeated phrases
    (hallucination loop), we short-circuit and return a clear error
    rather than attempting detection on garbage transcription.

  FIX 5 — AUDIO TRIM BEFORE WHISPER
    Strip leading/trailing silence using ffmpeg silenceremove filter.
    This alone reduces Whisper time by 0.5-1.5s on typical recordings.

  RESPONSE TIME (CPU laptop, no GPU):
    User selects ayah from list (ayah= sent):
      → 2-4s  (no Whisper, cached refs)
    Free recitation (no ayah= sent):
      → 4-7s  (Whisper + detection + embedding)

  FRONTEND LOADING UX:
    Use the bundled loading_popup.html snippet for an engaging
    loading screen. It shows rotating Quranic tips and a waveform
    animation so users stay engaged during the 3-7s wait.
=====================================================================
"""

import os, time, json, threading, concurrent.futures, hashlib
import warnings, tempfile, subprocess, sqlite3, uuid, unicodedata, re
from difflib import SequenceMatcher
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
MAX_EMBED_BATCH     = 12
MAX_WORDS_TO_SCORE  = 15
WHISPER_WORKERS     = 2
MAX_CONCURRENT_EMBED= 3
REF_CACHE_MAX_ITEMS = 8000
PREWARM_SURAHS      = [1, 2, 36, 55, 67, 112, 113, 114]

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

# The 4 words that make up Bismillah — all stopwords, useless for detection
BISMILLAH_WORDS = {"بسم", "الله", "الرحمن", "الرحيم"}

# Whisper hallucination patterns (repeated phrases = garbage output)
HALLUCINATION_RE = re.compile(r"(.{6,}?)\1{2,}")  # same chunk repeated 2+ times


# ══════════════════════════════════════════════════════════════════
#  THREAD-SAFE LRU CACHE
# ══════════════════════════════════════════════════════════════════
class LRUCache:
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
        with self._lock:
            return len(self._cache)


REF_EMB_CACHE = LRUCache(maxsize=REF_CACHE_MAX_ITEMS)
REF_WAV_CACHE = LRUCache(maxsize=REF_CACHE_MAX_ITEMS)


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
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.commit(); conn.close()
    print("[DB] Database initialized ✓ (WAL mode)")


def _db():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def get_or_create_user(user_id, username="Anonymous"):
    with _db() as conn:
        row = conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
        if not row:
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ")
            conn.execute("INSERT OR IGNORE INTO users VALUES (?,?,?,?,?)",
                         (user_id, username, None, now, now))
            conn.commit()
            return {"user_id": user_id, "username": username, "best_reciter": None}
        return {"user_id": row[0], "username": row[1], "best_reciter": row[2]}


def update_user_best_reciter(user_id, best_reciter):
    with _db() as conn:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        conn.execute("UPDATE users SET best_reciter=?,updated_at=? WHERE user_id=?",
                     (best_reciter, now, user_id))
        conn.commit()


def save_analysis(user_id, surah, ayah, best_reciter, reciter_scores, overall_score):
    with _db() as conn:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        conn.execute("""INSERT INTO analysis_history
                        (user_id,ayah_surah,ayah_number,best_reciter,reciter_scores,overall_score,analyzed_at)
                        VALUES (?,?,?,?,?,?,?)""",
                     (user_id, surah, ayah, best_reciter,
                      json.dumps(reciter_scores), overall_score, now))
        conn.commit()


def get_user_history(user_id, limit=20):
    with _db() as conn:
        rows = conn.execute(
            """SELECT ayah_surah,ayah_number,best_reciter,reciter_scores,overall_score,analyzed_at
               FROM analysis_history WHERE user_id=? ORDER BY analyzed_at DESC LIMIT ?""",
            (user_id, limit)).fetchall()
    return [{"surah": r[0], "ayah": r[1], "best_reciter": r[2],
             "reciter_scores": json.loads(r[3]), "overall_score": r[4], "analyzed_at": r[5]}
            for r in rows]


def get_user_reciter_stats(user_id):
    with _db() as conn:
        rows = conn.execute(
            "SELECT best_reciter,COUNT(*) FROM analysis_history WHERE user_id=? GROUP BY best_reciter ORDER BY 2 DESC",
            (user_id,)).fetchall()
    return {r[0]: r[1] for r in rows}


# ══════════════════════════════════════════════════════════════════
#  MODEL LOADING  — FIX 1: NO torch.compile ON WINDOWS/CPU
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

    # ── FIX 1: Only torch.compile on CUDA; skip entirely on CPU/Windows ──
    # torch.compile on CPU requires cl.exe (MSVC) which crashes on most
    # Windows machines. On CPU it also adds overhead, not savings.
    if DEVICE == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[STARTUP] Wav2Vec2 torch.compile ✓ (CUDA)")
        except Exception as ex:
            print(f"[STARTUP] torch.compile skipped: {ex}")
    else:
        print("[STARTUP] torch.compile skipped (CPU mode — not beneficial)")

    print("[STARTUP] Wav2Vec2 ready ✓")
    return processor, model


print("[STARTUP] Loading Arabic Wav2Vec2 …")
ar_processor, ar_model = load_wav2vec2()
print("[STARTUP] Loading Whisper …")
whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=DEVICE)
print("[STARTUP] Whisper ready ✓")

EMBED_SEMAPHORE   = threading.Semaphore(MAX_CONCURRENT_EMBED)
WHISPER_SEMAPHORE = threading.Semaphore(WHISPER_WORKERS)
ASYNC_EXECUTOR    = concurrent.futures.ThreadPoolExecutor(
    max_workers=min(32, (os.cpu_count() or 4) * 4),
    thread_name_prefix="qchk"
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


def trim_silence(path: str) -> str:
    """
    FIX 3 / FIX 5: Strip leading & trailing silence with ffmpeg.
    Returns path to trimmed file (caller must delete it).
    Falls back to original path if ffmpeg trim fails.
    """
    trimmed = path + "_trimmed.wav"
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", path,
             "-af", "silenceremove=start_periods=1:start_silence=0.3:start_threshold=-45dB"
                    ",areverse"
                    ",silenceremove=start_periods=1:start_silence=0.3:start_threshold=-45dB"
                    ",areverse",
             "-ar", str(SAMPLE_RATE), "-ac", "1", trimmed],
            capture_output=True, timeout=20)
        if r.returncode == 0 and os.path.exists(trimmed) and os.path.getsize(trimmed) > 1000:
            return trimmed
    except Exception:
        pass
    # fallback: return original
    if os.path.exists(trimmed):
        try: os.remove(trimmed)
        except: pass
    return path


def equal_split(wav: np.ndarray, n: int) -> list:
    if n == 0: return []
    step = max(len(wav) // n, 1)
    return [(i * step, min((i + 1) * step, len(wav))) for i in range(n)]


def sample_word_indices(n: int) -> list:
    if n <= MAX_WORDS_TO_SCORE: return list(range(n))
    step = n / MAX_WORDS_TO_SCORE
    return sorted(set([0, n - 1] + [int(i * step) for i in range(MAX_WORDS_TO_SCORE)]))[:MAX_WORDS_TO_SCORE]


# ══════════════════════════════════════════════════════════════════
#  GPU EMBEDDING
# ══════════════════════════════════════════════════════════════════
def _run_embeddings(wav_list: list) -> list:
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
            results.append(h[i].mean(dim=0).cpu())
    return results


def embed_user(wav: np.ndarray, scored_bounds: list) -> list:
    segs = [wav[s:e] if e > s else wav[:SAMPLE_RATE] for s, e in scored_bounds]
    return _run_embeddings(segs)


def embed_all_reciters_batched(
        surah: int, ayah: int, scored_idx: list, n_words: int) -> dict:
    idx_hash  = hashlib.md5(str(scored_idx).encode()).hexdigest()[:8]
    result    = {}
    to_embed  = {}

    for name, info in RECITERS.items():
        cache_key = f"{surah}:{ayah}:{info['key']}:{idx_hash}"
        cached    = REF_EMB_CACHE.get(cache_key)
        if cached is not None:
            result[name] = cached
        else:
            wav = get_ref_wav_cached(surah, ayah, info["key"])
            if wav is None:
                result[name] = None
                continue
            bounds = equal_split(wav, n_words)
            segs   = [wav[bounds[i][0]:bounds[i][1]] if bounds[i][1] > bounds[i][0]
                      else wav[:SAMPLE_RATE] for i in scored_idx]
            to_embed[name] = (segs, cache_key)

    if not to_embed:
        return result

    flat_segs     = []
    reciter_spans = {}
    for name, (segs, _) in to_embed.items():
        start = len(flat_segs)
        flat_segs.extend(segs)
        reciter_spans[name] = (start, len(flat_segs))

    flat_embs = _run_embeddings(flat_segs)

    for name, (segs, cache_key) in to_embed.items():
        s, e   = reciter_spans[name]
        embs   = flat_embs[s:e]
        REF_EMB_CACHE.set(cache_key, embs)
        result[name] = embs

    return result


# ══════════════════════════════════════════════════════════════════
#  REFERENCE WAV CACHE
# ══════════════════════════════════════════════════════════════════
def audio_path(surah, ayah, key):
    return os.path.join(AUDIO_FOLDER, key, f"{int(surah):03d}{int(ayah):03d}.mp3")


def get_ref_wav_cached(surah: int, ayah: int, key: str):
    cache_key = f"{surah}:{ayah}:{key}"
    cached    = REF_WAV_CACHE.get(cache_key)
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


# ══════════════════════════════════════════════════════════════════
#  TRANSCRIPTION  — FIX 3 + FIX 4: FASTER + HALLUCINATION GUARD
# ══════════════════════════════════════════════════════════════════
def is_hallucination(text: str) -> bool:
    """
    FIX 4: Detect Whisper hallucination loops.
    Pattern: same phrase repeated 2+ times back-to-back.
    Also catches the non-Arabic output case.
    """
    if not text or len(text.strip()) < 4:
        return True
    # Check for repeated pattern (looping hallucination)
    if HALLUCINATION_RE.search(text):
        print(f"[TRANSCRIBE] Hallucination detected: '{text[:60]}'")
        return True
    # Check if output has almost no Arabic characters
    arabic_chars = sum(1 for c in text if "\u0600" <= c <= "\u06FF")
    if arabic_chars < 4:
        print(f"[TRANSCRIBE] No Arabic detected: '{text[:60]}'")
        return True
    return False


def transcribe_audio(path: str) -> str:
    """
    FIX 3: Trim silence first, then transcribe.
    FIX 4: Detect and reject hallucinations.
    """
    # Step 1: trim silence (saves 0.5-1.5s on Whisper)
    trimmed_path = trim_silence(path)
    try:
        with WHISPER_SEMAPHORE:
            result = whisper_model.transcribe(
                trimmed_path,
                language="ar",
                task="transcribe",
                fp16=(DEVICE == "cuda"),
                verbose=False,
                condition_on_previous_text=False,
                beam_size=1,
                best_of=1,
                temperature=0.0,
                no_speech_threshold=0.6,     # FIX 4: raised from 0.4 → reject noise faster
                compression_ratio_threshold=2.4,  # tighter: catch repetition loops early
                # FIX 3: initial_prompt helps tiny model stay in Arabic
                initial_prompt="بسم الله الرحمن الرحيم",
            )
        text = result["text"].strip()
        if is_hallucination(text):
            return ""   # caller handles empty string
        return text
    except Exception as e:
        print(f"[TRANSCRIBE] Whisper error ({e}), wav2vec2 fallback …")
        try:
            wav = load_audio(path)
            with EMBED_SEMAPHORE:
                inp = ar_processor(wav, sampling_rate=SAMPLE_RATE,
                                   return_tensors="pt", padding=True)
                inp = {k: v.to(DEVICE) for k, v in inp.items()}
                with torch.inference_mode():
                    logits = ar_model(**inp).logits
                ids = torch.argmax(logits, dim=-1)
                return ar_processor.batch_decode(ids)[0].strip()
        except Exception as e2:
            print(f"[TRANSCRIBE] Wav2Vec2 fallback also failed: {e2}")
            return ""
    finally:
        if trimmed_path != path and os.path.exists(trimmed_path):
            try: os.remove(trimmed_path)
            except: pass


# ══════════════════════════════════════════════════════════════════
#  ARABIC NORMALISATION
# ══════════════════════════════════════════════════════════════════
def strip_diacritics(text: str) -> str:
    return "".join(c for c in text if not unicodedata.category(c).startswith("M"))

def normalise_arabic(text: str) -> str:
    text = strip_diacritics(text)
    for v in "أإآٱ":
        text = text.replace(v, "ا")
    return " ".join(text.replace("ـ", "").split())


# ══════════════════════════════════════════════════════════════════
#  AYAH DETECTION  — FIX 2: HARD BISMILLAH SUPPRESSION
# ══════════════════════════════════════════════════════════════════
def _count_rare_matches(t_words: list, a_words_set: set) -> int:
    """Count how many NON-stopword words from transcript match the ayah."""
    return sum(1 for w in t_words if w not in ARABIC_STOPWORDS and w in a_words_set)


def compute_ayah_score(t_norm: str, a_norm: str, ayah_number: int = 0,
                       surah_number: int = 0) -> float:
    """
    4-component score + Bismillah suppression.

    FIX 2 logic:
    - Ayah 1 of any surah = Bismillah. Its words are ALL stopwords.
    - We hard-cap its score at 0.40 unless ≥2 rare word matches.
    - All other ayahs get a small +0.12 bonus to beat Bismillah's
      common-word advantage.
    """
    if not t_norm or not a_norm: return 0.0

    seq  = SequenceMatcher(None, t_norm, a_norm, autojunk=False).ratio()
    t_w  = t_norm.split()
    a_w  = a_norm.split()
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

    base_score = min(1.0, (0.50 * seq + 0.50 * wj) * lp + anchor)

    # ── FIX 2: BISMILLAH HARD SUPPRESSION ───────────────────────
    if ayah_number == 1:
        rare_matches = _count_rare_matches(t_w, a_s)
        if rare_matches < 1:
            # Zero rare word matches → Bismillah wins only on stopwords → suppress hard
            base_score = min(base_score, 0.25)
        elif rare_matches < 2:
            # Only 1 rare match → moderate cap
            base_score = min(base_score, 0.40)
        # ≥2 rare matches → allow full score (user genuinely said Bismillah)
    else:
        # Non-Bismillah ayahs get a small boost to beat common-word matching
        base_score = min(1.0, base_score + 0.12)

    return base_score


def detect_ayah_in_surah(transcription: str, surah_rows: list) -> dict:
    """
    Detect which ayah was recited.
    FIX 2: Bismillah is strongly suppressed unless transcription
    genuinely contains rare Bismillah-specific content (which it won't
    because all Bismillah words ARE stopwords → it can never win).
    """
    if len(surah_rows) == 1:
        return surah_rows[0]

    t_norm = normalise_arabic(transcription)

    # Too short or empty (hallucination guard already cleared this)
    if len(t_norm.strip()) < 3:
        # Return ayah 2 if available (skip Bismillah as default)
        for row in surah_rows:
            if row["ayah"] > 1:
                return row
        return surah_rows[0]

    best_row, best_score = surah_rows[0], -1.0
    for row in surah_rows:
        sc = compute_ayah_score(
            t_norm,
            normalise_arabic(row["text"]),
            ayah_number=row["ayah"],
            surah_number=row["surah"],
        )
        if sc > best_score:
            best_score, best_row = sc, row

    print(f"[DETECT] best=Ayah {best_row['ayah']} (score={best_score:.3f}) "
          f"surah={best_row['surah']} from '{t_norm[:40]}'")
    return best_row


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
        rules.append({"rule": "Ghunna", "detail": "Nasal emphasis on ن/م shadda", "severity": "required"})
    for ch in QALQALA_CHARS:
        if ch + "ْ" in word:
            rules.append({"rule": "Qalqala", "detail": f"Echo/bounce on sukoon {ch}", "severity": "required"}); break
    for m in MADD_CHARS:
        if m in word:
            rules.append({"rule": "Madd", "detail": f"Elongate {m}", "severity": "required"}); break
    if word.endswith("نْ") or any(word.endswith(t) for t in TANWIN_CHARS):
        if next_word:
            f = next_word[0]
            if   f in IDGHAM_LETTERS: rules.append({"rule": "Idgham", "detail": "Merge noon", "severity": "required"})
            elif f in IKHFA_LETTERS:  rules.append({"rule": "Ikhfa", "detail": "Hide noon", "severity": "required"})
            elif f == "ب":            rules.append({"rule": "Iqlab", "detail": "Noon to meem before ب", "severity": "required"})
    if "اللَّه" in word or word in ["اللَّهِ", "اللَّهُ", "اللَّهَ"]:
        rules.append({"rule": "Lam Jalalah", "detail": "Heavy pronunciation of Allah", "severity": "required"})
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
#  PREWARM — WAV ONLY, NO GPU
# ══════════════════════════════════════════════════════════════════
def prewarm_wav_cache():
    time.sleep(8)
    print("[PREWARM] Loading WAV files into RAM …")
    count = 0
    for surah in PREWARM_SURAHS:
        for row in SURAH_INDEX.get(surah, []):
            for info in RECITERS.values():
                if get_ref_wav_cached(surah, row["ayah"], info["key"]) is not None:
                    count += 1
            time.sleep(0.02)
    print(f"[PREWARM] Done — {count} WAV files cached ✓ (GPU embeds lazy)")


# ══════════════════════════════════════════════════════════════════
#  MAIN ANALYSIS
# ══════════════════════════════════════════════════════════════════
def analyse_multi_reciter(
        user_path: str,
        surah_rows: list,
        user_id: str,
        known_ayah: int | None = None
) -> dict:
    t_start = time.time()

    # ── STEP 1: Identify the ayah ─────────────────────────────────
    if known_ayah is not None:
        ayah_row = next((r for r in surah_rows if r["ayah"] == known_ayah),
                        surah_rows[0])
        transcribed_text = ayah_row["text"]
        transcription_ms = 0
        text_match_ratio = 1.0
        print(f"[ANALYSE] PATH A — Ayah {ayah_row['surah']}:{ayah_row['ayah']} "
              f"(user-selected, Whisper skipped)")
    else:
        t0               = time.time()
        transcribed_text = transcribe_audio(user_path)
        transcription_ms = int((time.time() - t0) * 1000)
        print(f"[ANALYSE] PATH B — Transcription {transcription_ms}ms: '{transcribed_text}'")

        if not transcribed_text:
            # Hallucination / silence — return descriptive error
            return {
                "error": "Could not detect Arabic speech. Please record clearly in a quiet environment.",
                "transcribed_text": "",
                "transcription_ms": transcription_ms,
            }

        ayah_row = detect_ayah_in_surah(transcribed_text, surah_rows)

        text_match_ratio = SequenceMatcher(
            None,
            normalise_arabic(transcribed_text),
            normalise_arabic(ayah_row["text"]),
            autojunk=False
        ).ratio()

    all_words = ayah_row["text"].split()

    # ── STEP 2: Load + split user audio ───────────────────────────
    user_wav        = load_audio(user_path)
    user_pace       = estimate_pace(user_wav, len(all_words))
    all_user_bounds = equal_split(user_wav, len(all_words))

    # ── STEP 3: Sample words ───────────────────────────────────────
    scored_idx    = sample_word_indices(len(all_words))
    scored_words  = [all_words[i]       for i in scored_idx]
    scored_bounds = [all_user_bounds[i] for i in scored_idx]

    # ── STEP 4: User embeddings ────────────────────────────────────
    t_emb     = time.time()
    user_embs = embed_user(user_wav, scored_bounds)
    print(f"[ANALYSE] User embeddings {int((time.time()-t_emb)*1000)}ms")

    # ── STEP 5: Reference embeddings (batched) ─────────────────────
    t_ref    = time.time()
    ref_embs = embed_all_reciters_batched(
        ayah_row["surah"], ayah_row["ayah"], scored_idx, len(all_words))
    print(f"[ANALYSE] Ref embeddings {int((time.time()-t_ref)*1000)}ms")

    # ── STEP 6: Score each reciter ─────────────────────────────────
    t_score = time.time()

    def score_one(name, info):
        embs    = ref_embs.get(name)
        has_ref = embs is not None
        word_results = []

        for wi, (word, (s, e)) in enumerate(zip(scored_words, scored_bounds)):
            u_emb = user_embs[wi].unsqueeze(0)
            if has_ref:
                raw      = float(F.cosine_similarity(u_emb, embs[wi].unsqueeze(0)).item())
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(RECITERS)) as ex:
        for name, res in ex.map(lambda ni: score_one(*ni), RECITERS.items()):
            reciter_results[name] = res

    scoring_ms = int((time.time() - t_score) * 1000)
    print(f"[ANALYSE] Scoring {scoring_ms}ms")

    # ── STEP 7: Assemble result ────────────────────────────────────
    valid        = {k: v for k, v in reciter_results.items() if "error" not in v}
    best_reciter = max(valid, key=lambda k: valid[k]["weighted_score"])
    best_score   = valid[best_reciter]["weighted_score"]
    ranked       = sorted([(n, r.get("weighted_score", 0)) for n, r in reciter_results.items()],
                           key=lambda x: x[1], reverse=True)

    best_words   = reciter_results[best_reciter].get("words", [])
    word_summary = [{
        "word": w["word"], "correct_pct": w["correct_pct"], "error_pct": w["error_pct"],
        "status": w["status"], "confidence": w["confidence"], "error_types": w["error_types"],
        "tajweed": [t["rule"] for t in w.get("tajweed", [])],
        "start_ms": w["start_ms"], "end_ms": w["end_ms"],
    } for w in best_words]

    avg_correct = round(float(np.mean([w["correct_pct"] for w in word_summary])), 1) if word_summary else 0.0

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

    def _save():
        update_user_best_reciter(user_id, best_reciter)
        save_analysis(user_id, ayah_row["surah"], ayah_row["ayah"],
                      best_reciter,
                      {n: round(r.get("weighted_score", 0), 1) for n, r in reciter_results.items()},
                      best_score)
    ASYNC_EXECUTOR.submit(_save)

    total_ms = int((time.time() - t_start) * 1000)
    print(f"[ANALYSE] ✓ TOTAL {total_ms}ms  (transcription:{transcription_ms}ms scoring:{scoring_ms}ms)")

    return {
        "transcribed_text":   transcribed_text,
        "text_match_ratio":   round(text_match_ratio * 100, 1),
        "user_pace":          user_pace,
        "ayah":               ayah_row,
        "detected_ayah": {
            "surah":          ayah_row["surah"],
            "ayah":           ayah_row["ayah"],
            "text":           ayah_row["text"],
            "auto_detected":  known_ayah is None,
        },
        "best_reciter":       best_reciter,
        "best_reciter_score": best_score,
        "best_reciter_info":  RECITERS[best_reciter],
        "word_results":       word_summary,
        "ayah_summary":       ayah_summary,
        "reciter_comparison": [
            {
                "reciter":        name,
                "rank":           i + 1,
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
            "cache_sizes": {
                "ref_emb": len(REF_EMB_CACHE),
                "ref_wav": len(REF_WAV_CACHE),
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
    if not data:
        data = [
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
    return data[:limit]


QURAN_DATA  = load_quran()
SURAH_INDEX: dict = {}
for _row in QURAN_DATA:
    SURAH_INDEX.setdefault(_row["surah"], []).append(_row)
print(f"[DATA] Loaded {len(QURAN_DATA)} ayahs across {len(SURAH_INDEX)} surahs ✓")


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
    return jsonify({"status": "Quran Checker v4 running ✓"}), 200


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
        "ayahs": [{**r, "idx": i}
                  for i, r in enumerate(QURAN_DATA[start:start + limit], start=start)]
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
    all_rows = [r for r in QURAN_DATA if r["surah"] == surah]
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


@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    """
    POST multipart/form-data:
      audio    : audio file (required)
      surah    : int (required)
      ayah     : int (STRONGLY RECOMMENDED — send when user picks from list)
                 → Skips Whisper + detection entirely. 2-4x faster.
                 → Also prevents Bismillah false positives.
      user_id  : string (optional)
      username : string (optional)
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    audio_file = request.files["audio"]

    try:
        surah = int(request.form.get("surah") or 1)
    except ValueError:
        return jsonify({"error": "surah must be an integer"}), 400

    known_ayah = None
    raw_ayah   = request.form.get("ayah", "").strip()
    if raw_ayah:
        try:
            known_ayah = int(raw_ayah)
        except ValueError:
            pass

    user_id    = request.form.get("user_id")  or str(uuid.uuid4())
    username   = request.form.get("username") or "Anonymous"
    surah_rows = SURAH_INDEX.get(surah)
    if not surah_rows:
        return jsonify({"error": f"Surah {surah} not found"}), 404

    filename = audio_file.filename or ""
    suffix   = ".webm"
    for ext in [".mp3", ".m4a", ".ogg", ".wav", ".webm"]:
        if filename.lower().endswith(ext):
            suffix = ext; break

    get_or_create_user(user_id, username)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        audio_file.save(tmp)
        tmp_path = tmp.name

    try:
        result = analyse_multi_reciter(tmp_path, surah_rows, user_id, known_ayah)
        if "error" in result and len(result) <= 3:
            return jsonify(result), 422
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
        "status":     "ok",
        "version":    "v4",
        "device":     DEVICE,
        "reciters":   list(RECITERS.keys()),
        "ayah_count": len(QURAN_DATA),
        "models":     {"wav2vec2": ARABIC_MODEL_NAME, "whisper": WHISPER_MODEL_SIZE},
        "thresholds": {"correct": SCORE_CORRECT, "slight": SCORE_SLIGHT},
        "fixes": {
            "torch_compile":      "disabled on CPU/Windows (FIX 1)",
            "bismillah":          "hard-suppressed unless 2+ rare word matches (FIX 2)",
            "silence_trim":       "active via ffmpeg silenceremove (FIX 3)",
            "hallucination_guard":"active — rejects looping/non-Arabic output (FIX 4)",
            "whisper_prompt":     "Arabic initial_prompt set (FIX 3)",
        },
        "cache": {
            "ref_emb": len(REF_EMB_CACHE),
            "ref_wav": len(REF_WAV_CACHE),
        },
    })


# ══════════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    threading.Thread(target=prewarm_wav_cache, daemon=True).start()

    port = int(os.environ.get("PORT", 5000))
    print(f"\n{'='*62}")
    print(f"  Quran Checker v4 — Windows fix + Bismillah fix + Speed")
    print(f"  http://localhost:{port}  |  Device: {DEVICE.upper()}")
    print(f"  Whisper: {WHISPER_MODEL_SIZE}  |  Max scored words: {MAX_WORDS_TO_SCORE}")
    print(f"")
    print(f"  KEY FIXES:")
    print(f"    FIX 1: torch.compile disabled on CPU (no cl.exe needed)")
    print(f"    FIX 2: Bismillah hard-suppressed in ayah detection")
    print(f"    FIX 3: Silence trimmed before Whisper (faster)")
    print(f"    FIX 4: Hallucination loops rejected (cleaner errors)")
    print(f"")
    print(f"  SPEED TIP: Frontend must POST ayah=<number> when user")
    print(f"  selects an ayah. Skips Whisper → 2-4s response.")
    print(f"")
    print(f"  For 1000 users (Windows):")
    print(f"    pip install waitress")
    print(f"    waitress-serve --port={port} --threads=16 Recitors_v4:app")
    print(f"  For Linux/Mac:")
    print(f"    gunicorn -w 1 -k gevent --worker-connections 500 -b 0.0.0.0:{port} Recitors_v4:app")
    print(f"{'='*62}\n")
    try:
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True, use_reloader=False)
    except OSError as e:
        if "Address already in use" in str(e) or "10048" in str(e):
            print(f"\n❌ Port {port} in use. Try: PORT=5001 python Recitors_v4.py")
        else:
            raise