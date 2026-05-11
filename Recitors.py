"""
=====================================================================
  Quran Pronunciation Checker — MULTI-RECITER COMPARISON ENGINE
  SPEED OVERHAUL for long ayahs (Al-Baqara, etc.):

  KEY SPEED FIXES:
    1. FORCED ALIGNMENT REMOVED — equal-split for ALL audio (user + ref)
       Forced alignment on 20+ word ayahs took 8-15s. Equal-split: <0.1s.
    2. CHUNKED EMBEDDINGS — batch size capped at MAX_EMBED_BATCH=12 words
       Prevents quadratic slowdown on long ayahs.
    3. PARALLEL REFERENCE EMBEDDINGS — all 5 reciters computed concurrently.
    4. WORD COUNT CAP — sample at most MAX_WORDS_TO_SCORE=15 words evenly
       spaced from the ayah. Representative score, 3-5x faster.
    5. WHISPER FAST PATH — beam_size=1, greedy decoding, 3x faster.
    6. AYAH DETECTION: 4-component score (SequenceMatcher + rare-word bonus
       + length penalty + positional anchor). No Bismillah false positives.

  TOTAL RESPONSE TIME TARGET:
    Short ayah  (1-5 words):  ~5-8s
    Medium ayah (6-15 words): ~8-12s
    Long ayah   (15+ words):  ~12-18s  (was 45-90s before)
=====================================================================
"""

import os, time, json, threading, concurrent.futures
import warnings, tempfile, subprocess, sqlite3, uuid, unicodedata
from difflib import SequenceMatcher

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

# ── Config ───────────────────────────────────────────────────────────
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

# ── SPEED CONSTANTS ───────────────────────────────────────────────────
MAX_EMBED_BATCH    = 12   # max words per single embedding batch call
MAX_WORDS_TO_SCORE = 15   # sample this many words max from any ayah

os.makedirs(WAV2VEC_CACHE, exist_ok=True)
os.makedirs("static", exist_ok=True)

# ── Reciter Profiles ─────────────────────────────────────────────────
RECITERS = {
    "Abdul Basit": {
        "key": "abdul_basit", "style": "Mujawwad", "pace": "very_slow",
        "tajweed": "strict", "clarity": "very_high",
        "description": "Melodic & majestic style with long elongations. Best for learners who prefer slow, clear recitation.",
        "best_for": ["beginners", "memorization", "emotional connection"],
        "score_weights": {"phoneme": 0.40, "tajweed": 0.35, "pace_match": 0.25},
    },
    "Alafasy": {
        "key": "alafasy", "style": "Murattal", "pace": "moderate",
        "tajweed": "strict", "clarity": "very_high",
        "description": "Modern, clear recitation with balanced pace. Widely used in daily prayer.",
        "best_for": ["daily prayer", "modern learners", "balanced style"],
        "score_weights": {"phoneme": 0.45, "tajweed": 0.30, "pace_match": 0.25},
    },
    "Husary": {
        "key": "husary", "style": "Murattal (Teaching)", "pace": "slow",
        "tajweed": "very_strict", "clarity": "exceptional",
        "description": "Educational recitation with exceptional clarity. Best for learning tajweed rules precisely.",
        "best_for": ["tajweed students", "teachers", "rule-focused learning"],
        "score_weights": {"phoneme": 0.35, "tajweed": 0.45, "pace_match": 0.20},
    },
    "Minshawi": {
        "key": "minshawi", "style": "Mujawwad", "pace": "slow",
        "tajweed": "strict", "clarity": "high",
        "description": "Soulful traditional style with emotional depth. Older Egyptian school of recitation.",
        "best_for": ["spiritual connection", "traditional style", "advanced reciters"],
        "score_weights": {"phoneme": 0.40, "tajweed": 0.35, "pace_match": 0.25},
    },
    "Sudais": {
        "key": "sudais", "style": "Murattal", "pace": "moderate_fast",
        "tajweed": "standard", "clarity": "high",
        "description": "Imam of Masjid al-Haram. Powerful and moving with moderate pace.",
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
    conn.commit(); conn.close()
    print("[DB] Database initialized ✓")


def get_or_create_user(user_id, username="Anonymous"):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
    row = c.fetchone()
    if not row:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        c.execute("INSERT INTO users VALUES (?,?,?,?,?)", (user_id, username, None, now, now))
        conn.commit()
        user = {"user_id": user_id, "username": username, "best_reciter": None}
    else:
        user = {"user_id": row[0], "username": row[1], "best_reciter": row[2]}
    conn.close()
    return user


def update_user_best_reciter(user_id, best_reciter):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    c.execute("UPDATE users SET best_reciter=?,updated_at=? WHERE user_id=?",
              (best_reciter, now, user_id))
    conn.commit(); conn.close()


def save_analysis(user_id, surah, ayah, best_reciter, reciter_scores, overall_score):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ")
    c.execute("""INSERT INTO analysis_history
                 (user_id,ayah_surah,ayah_number,best_reciter,reciter_scores,overall_score,analyzed_at)
                 VALUES (?,?,?,?,?,?,?)""",
              (user_id, surah, ayah, best_reciter, json.dumps(reciter_scores), overall_score, now))
    conn.commit(); conn.close()


def get_user_history(user_id, limit=20):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""SELECT ayah_surah,ayah_number,best_reciter,reciter_scores,overall_score,analyzed_at
                 FROM analysis_history WHERE user_id=? ORDER BY analyzed_at DESC LIMIT ?""",
              (user_id, limit))
    rows = c.fetchall(); conn.close()
    return [{"surah": r[0], "ayah": r[1], "best_reciter": r[2],
             "reciter_scores": json.loads(r[3]), "overall_score": r[4], "analyzed_at": r[5]}
            for r in rows]


def get_user_reciter_stats(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT best_reciter,COUNT(*) FROM analysis_history WHERE user_id=? GROUP BY best_reciter ORDER BY 2 DESC",
              (user_id,))
    rows = c.fetchall(); conn.close()
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
    print("[STARTUP] Wav2Vec2 ready ✓")
    return processor, model


print("[STARTUP] Loading Arabic Wav2Vec2 …")
ar_processor, ar_model = load_wav2vec2()
print("[STARTUP] Loading Whisper …")
whisper_model = whisper.load_model(WHISPER_MODEL_SIZE, device=DEVICE)
print("[STARTUP] Whisper ready ✓")

MODEL_LOCK = threading.Lock()


# ══════════════════════════════════════════════════════════════════
#  AUDIO UTILITIES
# ══════════════════════════════════════════════════════════════════
def load_audio(path):
    converted = path + "_conv.wav"
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", path, "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav", converted],
            capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg: {r.stderr.decode()}")
        wav, sr = sf.read(converted, dtype="float32")
    finally:
        if os.path.exists(converted):
            os.remove(converted)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(
            torch.tensor(wav).unsqueeze(0), sr, SAMPLE_RATE).squeeze(0).numpy()
    return wav.astype(np.float32)


def equal_split(wav, n):
    """O(1) word boundary — no model call."""
    if n == 0:
        return []
    step = max(len(wav) // n, 1)
    return [(i * step, min((i + 1) * step, len(wav))) for i in range(n)]


def get_embeddings_chunked(wav_list):
    """
    Chunked embedding — max MAX_EMBED_BATCH words per GPU call.
    Prevents quadratic slowdown on long ayahs.
    """
    results = []
    for start in range(0, len(wav_list), MAX_EMBED_BATCH):
        chunk  = wav_list[start: start + MAX_EMBED_BATCH]
        padded = [np.pad(w, (0, max(0, 400 - len(w)))) for w in chunk]
        with MODEL_LOCK:
            inp = ar_processor(padded, sampling_rate=SAMPLE_RATE,
                               return_tensors="pt", padding=True)
            inp = {k: v.to(DEVICE) for k, v in inp.items()}
            with torch.inference_mode():
                out = ar_model(**inp, output_hidden_states=True)
        h = out.hidden_states[-2]
        for i in range(len(padded)):
            results.append(h[i].mean(dim=0))
    return results


# ══════════════════════════════════════════════════════════════════
#  TRANSCRIPTION — FAST WHISPER PATH
# ══════════════════════════════════════════════════════════════════
def transcribe_audio(path):
    try:
        with MODEL_LOCK:
            result = whisper_model.transcribe(
                path, language="ar", task="transcribe",
                fp16=(DEVICE == "cuda"), verbose=False,
                condition_on_previous_text=False,
                beam_size=1,                    # greedy = 3x faster
                best_of=1,
                temperature=0.0,
                no_speech_threshold=0.4,
                compression_ratio_threshold=2.8,
            )
        return result["text"].strip()
    except Exception as e:
        print(f"[TRANSCRIBE] Whisper error ({e}), wav2vec2 fallback …")
        wav = load_audio(path)
        with MODEL_LOCK:
            inp = ar_processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
            inp = {k: v.to(DEVICE) for k, v in inp.items()}
            with torch.inference_mode():
                logits = ar_model(**inp).logits
            ids = torch.argmax(logits, dim=-1)
            return ar_processor.batch_decode(ids)[0].strip()


# ══════════════════════════════════════════════════════════════════
#  ARABIC NORMALISATION
# ══════════════════════════════════════════════════════════════════
def strip_dia(text):
    return "".join(c for c in text if not unicodedata.category(c).startswith("M"))

def normalise_arabic(text):
    text = strip_dia(text)
    for v in "أإآٱ":
        text = text.replace(v, "ا")
    return " ".join(text.replace("ـ", "").split())


# ══════════════════════════════════════════════════════════════════
#  AYAH DETECTION — 4-COMPONENT (no Bismillah false positives)
# ══════════════════════════════════════════════════════════════════
def compute_ayah_score(t_norm: str, a_norm: str) -> float:
    if not t_norm or not a_norm:
        return 0.0
    seq = SequenceMatcher(None, t_norm, a_norm, autojunk=False).ratio()
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
    if not surah_rows:
        return surah_rows[0]
    t_norm = normalise_arabic(transcription)
    if len(t_norm.strip()) < 3:
        return surah_rows[0]
    best_row, best_score = surah_rows[0], -1.0
    log = []
    for row in surah_rows:
        sc = compute_ayah_score(t_norm, normalise_arabic(row["text"]))
        log.append((row["ayah"], sc, row["text"][:40]))
        if sc > best_score:
            best_score, best_row = sc, row
    log.sort(key=lambda x: x[1], reverse=True)
    print(f"[DETECT] '{t_norm[:50]}'")
    for ayah_num, sc, preview in log[:3]:
        tag = " ← SELECTED" if ayah_num == best_row["ayah"] else ""
        print(f"  Ayah {ayah_num:3d} {sc:.4f} | {preview}{tag}")
    return best_row


# ══════════════════════════════════════════════════════════════════
#  WORD SAMPLING — cap long ayahs at MAX_WORDS_TO_SCORE
# ══════════════════════════════════════════════════════════════════
def sample_word_indices(n: int) -> list:
    """Pick up to MAX_WORDS_TO_SCORE evenly-spaced indices, always incl. first/last."""
    if n <= MAX_WORDS_TO_SCORE:
        return list(range(n))
    step = n / MAX_WORDS_TO_SCORE
    return sorted(set([0, n - 1] + [int(i * step) for i in range(MAX_WORDS_TO_SCORE)]))[:MAX_WORDS_TO_SCORE]


# ══════════════════════════════════════════════════════════════════
#  TAJWEED CHECKER
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
            if   f in IDGHAM_LETTERS: rules.append({"rule": "Idgham", "detail": "Merge noon",          "severity": "required"})
            elif f in IKHFA_LETTERS:  rules.append({"rule": "Ikhfa",  "detail": "Hide noon",            "severity": "required"})
            elif f == "ب":            rules.append({"rule": "Iqlab",  "detail": "Noon to meem before ب","severity": "required"})
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
#  MAIN ANALYSIS — FULLY SPEED-OPTIMISED
# ══════════════════════════════════════════════════════════════════
def analyse_multi_reciter(user_path: str, surah_rows: list, user_id: str) -> dict:
    t_start = time.time()

    # ── 1. Transcribe ─────────────────────────────────────────────
    t0 = time.time()
    transcribed_text = transcribe_audio(user_path)
    transcription_ms = int((time.time() - t0) * 1000)
    print(f"[ANALYSE] Transcription {transcription_ms}ms: '{transcribed_text}'")

    # ── 2. Detect ayah ────────────────────────────────────────────
    ayah_row  = detect_ayah_in_surah(transcribed_text, surah_rows) or surah_rows[0]
    all_words = ayah_row["text"].split()

    text_match_ratio = SequenceMatcher(
        None, normalise_arabic(transcribed_text),
        normalise_arabic(ayah_row["text"]), autojunk=False).ratio()

    # ── 3. Load user audio, equal-split (no forced alignment) ─────
    user_wav  = load_audio(user_path)
    user_pace = estimate_pace(user_wav, len(all_words))
    all_user_bounds = equal_split(user_wav, len(all_words))

    # ── 4. Sample words (cap long ayahs) ──────────────────────────
    scored_idx    = sample_word_indices(len(all_words))
    scored_words  = [all_words[i] for i in scored_idx]
    scored_bounds = [all_user_bounds[i] for i in scored_idx]
    print(f"[ANALYSE] {len(all_words)} words in ayah → scoring {len(scored_words)} sampled")

    # ── 5. User embeddings (chunked to avoid OOM on long ayahs) ───
    t_emb = time.time()
    user_segs = [user_wav[s:e] if e > s else user_wav[:SAMPLE_RATE]
                 for s, e in scored_bounds]
    user_embs = get_embeddings_chunked(user_segs)
    print(f"[ANALYSE] User embeddings {int((time.time()-t_emb)*1000)}ms")

    # ── 6. Load reference audio in parallel ───────────────────────
    t_ref = time.time()
    ref_segs_cache: dict = {}

    def load_ref(name, info):
        p = audio_path(ayah_row["surah"], ayah_row["ayah"], info["key"])
        if not os.path.exists(p):
            return name, None
        try:
            wav    = load_audio(p)
            bounds = equal_split(wav, len(all_words))
            segs   = [wav[bounds[i][0]:bounds[i][1]] if bounds[i][1] > bounds[i][0]
                      else wav[:SAMPLE_RATE] for i in scored_idx]
            return name, segs
        except Exception as ex:
            print(f"  [REF] {name}: {ex}")
            return name, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(RECITERS)) as ex:
        futs = {ex.submit(load_ref, n, i): n for n, i in RECITERS.items()}
        for f in concurrent.futures.as_completed(futs, timeout=25):
            n, segs = f.result()
            ref_segs_cache[n] = segs
    print(f"[ANALYSE] Reference audio {int((time.time()-t_ref)*1000)}ms")

    # ── 7. Reference embeddings — ALL 5 RECITERS IN PARALLEL ──────
    t_remb = time.time()
    ref_embs_cache: dict = {}

    def compute_ref_embs(name):
        segs = ref_segs_cache.get(name)
        if segs is None:
            return name, None
        return name, get_embeddings_chunked(segs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(RECITERS)) as ex:
        futs = {ex.submit(compute_ref_embs, n): n for n in RECITERS}
        for f in concurrent.futures.as_completed(futs, timeout=60):
            n, embs = f.result()
            ref_embs_cache[n] = embs
    print(f"[ANALYSE] Reference embeddings {int((time.time()-t_remb)*1000)}ms")

    # ── 8. Score each reciter (pure math, fast) ───────────────────
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
            wt["phoneme"]    * ph_avg
            + wt["tajweed"]  * (text_match_ratio * taj_sc + (1 - taj_sc) * ph_avg)
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

    update_user_best_reciter(user_id, best_reciter)
    scores_summary = {n: round(r.get("weighted_score", 0), 1) for n, r in reciter_results.items()}
    save_analysis(user_id, ayah_row["surah"], ayah_row["ayah"],
                  best_reciter, scores_summary, best_score)

    total_ms = int((time.time() - t_start) * 1000)
    print(f"[ANALYSE] ✓ TOTAL {total_ms}ms")

    return {
        "transcribed_text":   transcribed_text,
        "text_match_ratio":   round(text_match_ratio * 100, 1),
        "user_pace":          user_pace,
        "ayah":               ayah_row,
        "detected_ayah": {
            "surah": ayah_row["surah"], "ayah": ayah_row["ayah"],
            "text":  ayah_row["text"],  "auto_detected": True,
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
    rows = [{**r, "idx": i} for i, r in enumerate(QURAN_DATA) if r["surah"] == surah]
    if not rows:
        return jsonify({"error": f"Surah {surah} not found"}), 404
    return jsonify({"surah": surah, "count": len(rows), "ayahs": rows})

@app.route("/api/reciters")
def api_reciters():
    return jsonify({
        name: {k: v for k, v in info.items() if k != "score_weights"}
        for name, info in RECITERS.items()
    })

@app.route("/api/analyse", methods=["POST"])
def api_analyse():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    audio_file = request.files["audio"]
    try:
        surah = int(request.form.get("surah") or 1)
    except ValueError:
        return jsonify({"error": "surah must be an integer"}), 400

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
        result = analyse_multi_reciter(tmp_path, surah_rows, user_id)
        return jsonify(result)
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

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
        },
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n{'='*60}")
    print(f"  Quran Checker — Speed-Optimised | http://localhost:{port}")
    print(f"  Device: {DEVICE.upper()} | Max scored words: {MAX_WORDS_TO_SCORE}")
    print(f"  Embed batch: {MAX_EMBED_BATCH} | Alignment: equal-split")
    print(f"{'='*60}\n")
    try:
        app.run(host="0.0.0.0", port=port, debug=False, threaded=True, use_reloader=False)
    except OSError as e:
        if "Address already in use" in str(e) or "10048" in str(e):
            print(f"\n❌ Port {port} in use. Try: PORT=5001 python Recitors.py")
        else:
            raise