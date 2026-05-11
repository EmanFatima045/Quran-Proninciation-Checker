# 🕌 Quran Pronunciation Checker (AI Backend)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black)
![AI](https://img.shields.io/badge/AI-Deep%20Learning-orange)
![CUDA](https://img.shields.io/badge/CUDA-GPU-green)
![Whisper](https://img.shields.io/badge/Whisper-ASR-purple)
![PyTorch](https://img.shields.io/badge/PyTorch-Framework-red)
![HuggingFace](https://img.shields.io/badge/Transformers-HuggingFace-yellow)

A production-grade AI backend system for analyzing Quranic recitation, detecting pronunciation accuracy, and comparing user audio with multiple renowned Quran reciters.

It combines **Whisper**, **Wav2Vec2**, and **deep embedding similarity scoring** with **Tajweed-aware evaluation rules** to provide detailed, word-level feedback on Quran recitation quality.

---

## 🚀 Key Features

* 🎙️ Arabic Speech-to-Text using OpenAI Whisper (optimized inference)
* 🧠 Fallback transcription using Wav2Vec2 Arabic model
* 📖 Auto-detection of recited Ayah from audio
* 👥 Multi-reciter comparison engine (5+ reciters)
* 🔤 Word-by-word pronunciation scoring
* 📊 Weighted scoring (phoneme + tajweed + pace)
* ⚡ High-performance parallel processing (ThreadPoolExecutor)
* 🧩 Chunked embedding pipeline (GPU optimized)
* 💾 SQLite-based user history tracking
* 🔄 Fast equal-split alignment for long Surahs

---

## 🧠 AI / ML Architecture

### Models Used

| Component          | Technology                  |
| ------------------ | --------------------------- |
| Speech Recognition | Whisper (OpenAI)            |
| Arabic Embeddings  | Wav2Vec2 (HuggingFace)      |
| Similarity Metric  | Cosine Similarity           |
| Scoring Engine     | Custom Weighted Model       |
| Tajweed Rules      | Rule-based heuristic system |

---

## ⚙️ Tech Stack

* Python 3.10+
* Flask
* PyTorch
* HuggingFace Transformers
* Librosa / SoundFile
* FFmpeg
* SQLite

---

## 📁 Project Structure

```
backend/
│
├── Recitors.py          # Core AI engine (analysis pipeline)
├── app.py               # Flask API server
├── cache/               # Cached Wav2Vec2 models
├── data/                # Quran text dataset
├── uploads/             # User audio storage
├── database.db          # SQLite database
└── utils/               # Helper functions
```

---

## 🌐 API Endpoints

### 🎤 Analyze Recitation

```
POST /analyze
```

**Request:**

```json
{
  "audio": "file.wav",
  "user_id": "user123",
  "surah": 1
}
```

**Response:**

```json
{
  "best_reciter": "Al-Sudais",
  "best_reciter_score": 92.4,
  "transcribed_text": "...",
  "detected_ayah": {
    "surah": 1,
    "ayah": 1,
    "text": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
  },
  "word_results": [...],
  "reciter_comparison": [...]
}
```

---

### 🎧 Get Reciter Audio

```
GET /audio/<reciter>/<surah><ayah>.mp3
```

Example:

```
/audio/sudais/001001.mp3
```

---

## ⚡ Performance Optimizations

This backend is optimized for large Surahs like **Al-Baqarah**.

### 🚀 Optimizations Applied

* ❌ Removed forced alignment (slow step)
* ⚡ Replaced with equal-split segmentation (O(1) operation)
* 🧠 Chunked embedding inference (GPU memory safe)
* 🔄 Parallel processing for reciters
* 🎯 Whisper optimized decoding:

  * beam_size = 1
  * best_of = 1
  * temperature = 0.0
* 💾 Model caching (Wav2Vec2 local storage)

### ⏱ Performance Result

| Version          | Time per Ayah   |
| ---------------- | --------------- |
| Old System       | 8–15 seconds    |
| Optimized System | **2–3 seconds** |

---

## 📊 Scoring System

Final reciter score is computed using:

```
Final Score =
(0.45 × Phoneme Similarity)
+ (0.30 × Tajweed Accuracy)
+ (0.25 × Pace Match)
```

### Word-Level Analysis

Each word is classified as:

* ✅ Correct (≥ 82%)
* ⚠️ Slight Error (55–81%)
* ❌ Incorrect (< 55%)

---

## 👥 Supported Reciters

* Mishary Rashid Alafasy
* Abdul Basit Abdul Samad
* Al-Sudais
* Al-Husary
* Al-Minshawi

Each reciter includes:

* Style profile
* Pace classification
* Tajweed emphasis weights

---

## 🗄 Database Schema

### Users Table

* user_id (TEXT)
* username (TEXT)
* best_reciter (TEXT)
* created_at (TEXT)
* updated_at (TEXT)

### Analysis History

* id (INTEGER)
* user_id (TEXT)
* surah (INTEGER)
* ayah (INTEGER)
* best_reciter (TEXT)
* scores (JSON)
* timestamp (TEXT)

---

## ▶️ Installation & Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Install FFmpeg

**Windows:** Download from [https://ffmpeg.org](https://ffmpeg.org)

**Linux:**

```bash
sudo apt install ffmpeg
```

---

### 3. Run Backend

```bash
python Recitors.py
```

Server will start at:

```
http://localhost:5000
```

---

## 🔐 Environment Variables

```
PORT=5000
DEVICE=cuda  # or cpu
WHISPER_MODEL_SIZE=base
```

---

## 🧪 Future Improvements

* 🔴 Real-time streaming analysis
* 🎧 WebSocket live feedback system
* 📱 Mobile app integration
* 🧠 AI pronunciation correction suggestions
* 📊 Tajweed visualization engine

---

## 👩‍💻 Author

**Eman Fatima**

AI Engineer | Deep Learning | NLP | Speech Processing

---

## 📜 License

This project is intended for educational and research purposes.
