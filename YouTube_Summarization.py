import streamlit as st
import re
import time
import tempfile
import os
import multiprocessing
import subprocess
import shutil
import json
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
from collections import Counter

# YouTube transcript
from youtube_transcript_api import YouTubeTranscriptApi, VideoUnavailable, NoTranscriptFound

# LLM: prefer new langchain-ollama, fallback to deprecated
try:
    from langchain_ollama import OllamaLLM as Ollama
    OLLAMA_NEW = True
except ImportError:
    from langchain_community.llms import Ollama
    OLLAMA_NEW = False
    st.warning("For better performance, install langchain-ollama: `pip install -U langchain-ollama`", icon="⚠️")

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Audio transcription availability flags
WHISPER_AVAILABLE = False
FASTER_WHISPER_AVAILABLE = False

# Try faster-whisper first
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    try:
        import whisper
        WHISPER_AVAILABLE = True
    except ImportError:
        pass

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False

# Evaluation: ROUGE
from rouge_score import rouge_scorer

# Evaluation: BERTScore
BERTSCOPE_AVAILABLE = False
try:
    from bert_score import score as bert_score
    BERTSCOPE_AVAILABLE = True
except ImportError:
    pass

# NLP: keyword extraction, sentiment, NER, wordcloud, plotting, QA, embeddings, topic modeling
KEYBERT_AVAILABLE = False
YAKE_AVAILABLE = False
try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError:
    try:
        import yake
        YAKE_AVAILABLE = True
    except ImportError:
        pass

TEXTBLOB_AVAILABLE = False
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    pass

SPACY_AVAILABLE = False
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
        SPACY_AVAILABLE = True
    except OSError:
        st.warning("spaCy small model not found. Run: python -m spacy download en_core_web_sm")
except ImportError:
    pass

WORDCLOUD_AVAILABLE = False
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    WORDCLOUD_AVAILABLE = True
except ImportError:
    pass

PLOTLY_AVAILABLE = False
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    pass

TRANSFORMERS_AVAILABLE = False
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

SENTENCE_TRANSFORMERS_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    pass

SKLEARN_AVAILABLE = False
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    SKLEARN_AVAILABLE = True
except ImportError:
    pass

# ---------------------------
# Data structures
# ---------------------------
@dataclass
class Segment:
    """Represents a transcript segment with text and timing."""
    text: str
    start: float
    end: float

@dataclass
class Chunk:
    """A chunk of text with its time range."""
    text: str
    start: float
    end: float
    index: int

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="🎥 Ultimate YouTube Summarizer", page_icon="🎥", layout="wide")
st.title("🎥 Ultimate YouTube Summarizer (Local LLM + Whisper + BERTScore)")
st.markdown("No API keys. Summarises any video – with or without captions. Everything runs **locally** on your machine.")

# ---------------------------
# HELPER FUNCTIONS
# ---------------------------
def get_cpu_count():
    try:
        return multiprocessing.cpu_count()
    except:
        return 2

@st.cache_data(show_spinner=False)
def get_video_metadata(video_id: str) -> Tuple[Optional[int], Optional[str]]:
    if not YT_DLP_AVAILABLE:
        return None, None
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://youtube.com/watch?v={video_id}", download=False)
            return info.get('duration'), info.get('title')
    except Exception as e:
        st.warning(f"Could not fetch metadata: {e}")
        return None, None

@st.cache_data(show_spinner=False)
def fetch_transcript_with_timestamps(video_id: str) -> Tuple[Optional[List[Segment]], str]:
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        segments = []
        for item in transcript_list:
            start = item['start']
            duration = item.get('duration', 0)
            end = start + duration
            segments.append(Segment(text=item['text'], start=start, end=end))
        return segments, "youtube_captions"
    except Exception as e:
        return None, str(e)

def split_audio_file(input_path: str, segment_minutes: int, output_dir: str) -> List[str]:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg not found. Please install ffmpeg to enable audio splitting.")
    segment_seconds = segment_minutes * 60
    basename = os.path.splitext(os.path.basename(input_path))[0]
    pattern = os.path.join(output_dir, f"{basename}_%03d.mp3")
    cmd = [
        "ffmpeg", "-i", input_path, "-f", "segment", "-segment_time", str(segment_seconds),
        "-c", "copy", "-map", "0", pattern
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    segments = sorted([f for f in os.listdir(output_dir) if f.startswith(basename) and f.endswith(".mp3")])
    return [os.path.join(output_dir, f) for f in segments]

def transcribe_audio_segment_faster(segment_path: str, model) -> List[Segment]:
    segments, info = model.transcribe(segment_path, word_timestamps=False)
    result = []
    for seg in segments:
        result.append(Segment(
            text=seg.text.strip(),
            start=seg.start,
            end=seg.end
        ))
    return result

def transcribe_audio_segment_whisper(segment_path: str, model) -> List[Segment]:
    result = model.transcribe(segment_path, word_timestamps=False)
    segs = []
    for seg in result['segments']:
        segs.append(Segment(
            text=seg['text'].strip(),
            start=seg['start'],
            end=seg['end']
        ))
    return segs

# ==================== IMPROVED TRANSCRIBE FUNCTION (WITH COOKIE FALLBACK) ====================
def transcribe_audio_with_timestamps(
    video_id: str,
    whisper_model: str,
    cookies_file: Optional[str],
    max_minutes: Optional[int],
    split_minutes: int,
    use_faster_whisper: bool = False,
    use_browser_cookies: bool = False,
    browser_name: str = "chrome",
    max_retries: int = 3,
    retry_delay: int = 5
) -> Tuple[Optional[List[Segment]], str]:
    """
    Download audio and transcribe with Whisper.
    Includes retry logic, fallback formats, and a live progress bar.
    """
    if not (YT_DLP_AVAILABLE and (FASTER_WHISPER_AVAILABLE or WHISPER_AVAILABLE)):
        return None, "Missing dependencies (yt-dlp and a Whisper backend)"

    # Create a progress bar and status text
    progress_bar = st.progress(0, text="Initializing...")
    status_text = st.empty()

    def update_progress(percent: float, message: str):
        progress_bar.progress(min(percent, 1.0), text=message)
        status_text.text(message)

    temp_dir = tempfile.mkdtemp()
    output_template = os.path.join(temp_dir, "%(title)s.%(ext)s")

    # Base options – increased timeout, retries, and multiple clients
    base_ydl_opts = {
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_template,
        'quiet': True,
        'no_warnings': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'geo_bypass': True,
        'geo_bypass_country': 'US',
        'extractor_retries': 5,
        'file_access_retries': 5,
        'fragment_retries': 5,
        'age_limit': None,
        'extractor_args': {'youtube': {'player_client': ['android', 'web', 'ios']}},
        'socket_timeout': 30,
        'retries': 10,
    }

    # List of cookie options to try (browser, file, none)
    cookie_options = []
    if use_browser_cookies:
        cookie_options.append(('browser', {'cookiesfrombrowser': (browser_name,)}))
    if cookies_file and os.path.exists(cookies_file):
        cookie_options.append(('file', {'cookiefile': cookies_file}))
    cookie_options.append(('none', {}))   # always try no cookies as a fallback

    # Formats to try in order of preference
    formats_to_try = [
        'bestaudio[ext=m4a]/bestaudio/best',   # primary
        'worstaudio/worst',                      # fallback 1
        'bestaudio[ext=webm]/bestaudio',         # fallback 2
    ]

    filename = None
    update_progress(0.05, "Attempting to download audio...")

    # Outer loop: retry attempts (overall)
    for attempt in range(max_retries):
        # For each attempt, try all cookie options
        for cookie_name, cookie_opts in cookie_options:
            # For each cookie option, try all formats
            for fmt_idx, fmt in enumerate(formats_to_try):
                # Build ydl_opts for this combination
                ydl_opts = base_ydl_opts.copy()
                ydl_opts.update(cookie_opts)   # add cookies if any
                ydl_opts['format'] = fmt

                msg = f"Attempt {attempt+1}/{max_retries} | Cookies: {cookie_name} | Format {fmt_idx+1}/{len(formats_to_try)}"
                update_progress(0.1 + attempt * 0.1, msg)

                try:
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(f"https://youtube.com/watch?v={video_id}", download=True)
                        filename = ydl.prepare_filename(info)
                        base, ext = os.path.splitext(filename)
                        if ext not in ['.mp3', '.m4a', '.webm']:
                            possible = base + '.mp3'
                            if os.path.exists(possible):
                                filename = possible

                    if os.path.exists(filename):
                        update_progress(0.4, "Download complete.")
                        # Break out of all loops – we have a file
                        raise StopIteration
                    filename = None
                except StopIteration:
                    # Success – break all loops
                    break
                except Exception as e:
                    error_str = str(e).lower()
                    # If the error is cookie-related, skip this cookie option and continue to next one
                    if "cookie" in error_str or "database" in error_str:
                        st.warning(f"Cookie method '{cookie_name}' failed: {e}. Trying next cookie method.")
                        # This cookie option failed; go to next cookie option (continue outer cookie loop)
                        break
                    else:
                        # Non-cookie error, log and try next format
                        st.warning(f"Download attempt {attempt+1} with format {fmt} (cookies={cookie_name}) failed: {e}")
                        # Continue to next format
                        continue
            else:
                # This cookie option exhausted all formats without success; continue to next cookie option
                continue
            # If we reached here, we broke out of the inner loop because of success
            break
        else:
            # No cookie option succeeded in this attempt
            if attempt < max_retries - 1:
                update_progress(0.2 + attempt * 0.1, f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                update_progress(1.0, "Download failed after all attempts.")
                progress_bar.empty()
                status_text.empty()
                return None, f"Failed to download audio after {max_retries} attempts."
            continue
        break   # success – exit outer retry loop

    if not os.path.exists(filename):
        files = [f for f in os.listdir(temp_dir) if f.endswith('.mp3')]
        if files:
            filename = os.path.join(temp_dir, files[0])
        else:
            update_progress(1.0, "No audio file found.")
            progress_bar.empty()
            status_text.empty()
            return None, "No audio file produced after download."

    # Truncate if requested
    if max_minutes and max_minutes > 0:
        update_progress(0.45, f"Truncating to first {max_minutes} minutes...")
        truncated_filename = os.path.join(temp_dir, "truncated.mp3")
        cmd = ["ffmpeg", "-i", filename, "-t", str(max_minutes*60), "-c", "copy", truncated_filename]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        filename = truncated_filename

    # Transcription
    try:
        if use_faster_whisper and FASTER_WHISPER_AVAILABLE:
            model = WhisperModel(whisper_model, device="cpu", compute_type="int8")
            transcribe_func = transcribe_audio_segment_faster
        else:
            model = whisper.load_model(whisper_model)
            transcribe_func = transcribe_audio_segment_whisper

        if split_minutes <= 0:
            update_progress(0.5, "Transcribing audio (this may take a while)...")
            segs = transcribe_func(filename, model)
            update_progress(1.0, "Transcription complete.")
            progress_bar.empty()
            status_text.empty()
            return segs, "whisper_transcription"
        else:
            update_progress(0.5, f"Splitting audio into {split_minutes}-minute segments...")
            segment_files = split_audio_file(filename, split_minutes, temp_dir)
            segment_files.sort()
            total_segments = len(segment_files)
            all_segments = []
            completed = 0

            with ThreadPoolExecutor(max_workers=min(total_segments, get_cpu_count())) as executor:
                future_to_file = {executor.submit(transcribe_func, f, model): f for f in segment_files}
                results = []
                for future in as_completed(future_to_file):
                    fname = future_to_file[future]
                    try:
                        segs = future.result()
                        results.append((fname, segs))
                    except Exception as e:
                        st.warning(f"Segment {fname} failed: {e}")
                    completed += 1
                    update_progress(0.5 + 0.5 * (completed / total_segments),
                                    f"Transcribed {completed}/{total_segments} segments")

                results.sort(key=lambda x: x[0])
                for fname, segs in results:
                    try:
                        base_num = int(fname.split('_')[-1].split('.')[0])
                        offset = base_num * split_minutes * 60
                    except:
                        offset = 0.0
                    for s in segs:
                        s.start += offset
                        s.end += offset
                    all_segments.extend(segs)

            update_progress(1.0, "Transcription complete.")
            progress_bar.empty()
            status_text.empty()
            return all_segments, "whisper_transcription_split"
    except Exception as e:
        error_msg = str(e)
        update_progress(1.0, "Transcription failed.")
        progress_bar.empty()
        status_text.empty()
        return None, f"Transcription failed: {error_msg}"
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass
# ==================== END OF IMPROVED FUNCTION ====================

def extract_video_id(url: str) -> Optional[str]:
    patterns = [
        r"(?:youtu\.be/)([0-9A-Za-z_-]{11})",
        r"(?:youtube\.com/watch\?v=)([0-9A-Za-z_-]{11})",
        r"(?:youtube\.com/embed/)([0-9A-Za-z_-]{11})",
        r"(?:youtube\.com/v/)([0-9A-Za-z_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    if "v=" in url:
        return url.split("v=")[-1].split("&")[0]
    elif "/" in url:
        candidate = url.split("/")[-1].split("?")[0]
        if len(candidate) == 11:
            return candidate
    return None

def segments_to_text(segments: List[Segment]) -> str:
    return " ".join([s.text for s in segments])

def chunk_segments(segments: List[Segment], chunk_size: int = 1200, overlap: int = 200) -> List[Chunk]:
    if not segments:
        return []
    chunks = []
    current_segments = []
    current_text = ""
    current_start = segments[0].start
    current_end = segments[0].end
    for seg in segments:
        if len(current_text) + len(seg.text) > chunk_size and current_text:
            chunk_text = " ".join([s.text for s in current_segments])
            chunks.append(Chunk(text=chunk_text, start=current_start, end=current_end, index=len(chunks)))
            current_segments = [seg]
            current_text = seg.text
            current_start = seg.start
            current_end = seg.end
        else:
            current_segments.append(seg)
            current_text += " " + seg.text if current_text else seg.text
            current_end = seg.end
    if current_segments:
        chunks.append(Chunk(text=" ".join([s.text for s in current_segments]), start=current_start, end=current_end, index=len(chunks)))
    return chunks

def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    else:
        return f"{m:02d}:{s:02d}"

def summarize_chunk(chunk: Chunk, llm_chain, include_timestamp: bool = True, timeout: int = 60) -> Optional[str]:
    try:
        summary = llm_chain.invoke({"text": chunk.text})
        if include_timestamp:
            time_range = f"[{format_timestamp(chunk.start)} - {format_timestamp(chunk.end)}]"
            return f"{time_range} {summary}"
        else:
            return summary
    except Exception:
        return None

def final_summary(chunk_summaries: List[str], llm, custom_prompt: Optional[str] = None) -> str:
    combined = "\n".join(chunk_summaries)
    if custom_prompt:
        prompt = PromptTemplate.from_template(custom_prompt + "\n\n{text}\n\nSummary:")
    else:
        prompt = PromptTemplate.from_template("""
You are an expert summarizer.

Here are timestamped summaries of video segments:
{text}

Create a structured executive summary of the entire video.
Include key themes and insights. Do not add information not present in the summaries.

Final Summary:
""")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"text": combined})

def evaluate_rouge(reference: str, summary: str) -> Optional[float]:
    if not reference or not summary:
        return None
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, summary)
    return scores['rougeL'].fmeasure

def evaluate_bertscore(reference: str, summary: str, model_type: str = "roberta-large") -> Optional[float]:
    if not BERTSCOPE_AVAILABLE:
        return None
    try:
        P, R, F1 = bert_score([summary], [reference], lang="en", model_type=model_type, verbose=False)
        return F1.item()
    except Exception as e:
        st.warning(f"BERTScore computation failed: {e}")
        return None

def verify_summary(transcript: str, summary: str, llm) -> str:
    prompt = PromptTemplate.from_template("""
Compare the transcript and summary.

Remove any statement not explicitly supported.
Rewrite corrected summary.

Transcript:
{transcript}

Summary:
{summary}

Corrected Summary:
""")
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"transcript": transcript[:4000], "summary": summary})

def format_transcription(segments: List[Segment]) -> str:
    lines = []
    for seg in segments:
        lines.append(f"[{format_timestamp(seg.start)}] {seg.text}")
    return "\n".join(lines)

def filter_chunks_by_strategy(chunks: List[Chunk], strategy: str, sample_rate: int, max_chunks: int) -> List[Chunk]:
    if strategy == "All chunks":
        return chunks
    elif strategy == "Sample every N chunks":
        return [chunks[i] for i in range(0, len(chunks), sample_rate)]
    elif strategy == "First N chunks only":
        return chunks[:max_chunks]
    else:
        return chunks

# ---------------------------
# NLP LEVEL 1: LEXICAL ANALYSIS
# ---------------------------
def lexical_analysis_keywords(text: str, top_n: int = 10) -> List[str]:
    if KEYBERT_AVAILABLE:
        try:
            kw_model = KeyBERT()
            keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=top_n)
            return [kw for kw, score in keywords]
        except:
            pass
    if YAKE_AVAILABLE:
        try:
            kw_extractor = yake.KeywordExtractor()
            keywords = kw_extractor.extract_keywords(text)
            return [kw for kw, score in keywords[:top_n]]
        except:
            pass
    return []

def lexical_analysis_wordfreq(text: str, top_n: int = 10) -> List[Tuple[str, int]]:
    words = re.findall(r'\b\w+\b', text.lower())
    stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    words = [w for w in words if w not in stopwords and len(w) > 2]
    return Counter(words).most_common(top_n)

def lexical_analysis_postags(text: str, sample_size: int = 500) -> Dict[str, int]:
    if not SPACY_AVAILABLE:
        return {}
    doc = nlp(text[:10000])
    pos_counts = Counter([token.pos_ for token in doc])
    return dict(pos_counts.most_common(10))

# ---------------------------
# NLP LEVEL 2: SYNTACTIC ANALYSIS
# ---------------------------
def syntactic_analysis_phrases(text: str, max_phrases: int = 10) -> Dict[str, List[str]]:
    if not SPACY_AVAILABLE:
        return {}
    doc = nlp(text[:20000])
    noun_phrases = [chunk.text for chunk in doc.noun_chunks][:max_phrases]
    verb_phrases = []
    for token in doc:
        if token.pos_ == "VERB" and len(verb_phrases) < max_phrases:
            phrase = " ".join([child.text for child in token.subtree])
            verb_phrases.append(phrase)
    return {"noun_phrases": noun_phrases, "verb_phrases": verb_phrases}

def syntactic_analysis_dependencies(text: str, top_relations: int = 5) -> Dict[str, int]:
    if not SPACY_AVAILABLE:
        return {}
    doc = nlp(text[:20000])
    dep_counts = Counter([token.dep_ for token in doc])
    return dict(dep_counts.most_common(top_relations))

# ---------------------------
# NLP LEVEL 3: SEMANTIC ANALYSIS
# ---------------------------
def semantic_sentiment_timeline(segments: List[Segment]) -> List[Dict]:
    if not TEXTBLOB_AVAILABLE:
        return []
    timeline = []
    for seg in segments:
        blob = TextBlob(seg.text)
        timeline.append({
            'start': seg.start,
            'end': seg.end,
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        })
    return timeline

def semantic_extract_entities(text: str) -> Dict[str, List[str]]:
    if not SPACY_AVAILABLE:
        return {}
    doc = nlp(text[:100000])
    entities = {}
    for ent in doc.ents:
        if ent.label_ not in entities:
            entities[ent.label_] = []
        if ent.text not in entities[ent.label_]:
            entities[ent.label_].append(ent.text)
    return entities

def semantic_key_quotes(segments: List[Segment], top_n: int = 5) -> List[Dict]:
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        scored = sorted(segments, key=lambda s: len(s.text), reverse=True)
        return [{'text': s.text, 'start': s.start, 'end': s.end, 'score': len(s.text)} for s in scored[:top_n]]
    
    sentences = [seg.text for seg in segments]
    if not sentences:
        return []
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sentences, convert_to_tensor=True)
    centroid = embeddings.mean(axis=0)
    scores = util.cos_sim(embeddings, centroid).squeeze().cpu().tolist()
    ranked = sorted(zip(scores, segments), key=lambda x: x[0], reverse=True)
    selected = []
    for score, seg in ranked:
        if len(selected) >= top_n:
            break
        if selected:
            seg_emb = embeddings[segments.index(seg)]
            max_sim = max(util.cos_sim(seg_emb, embeddings[selected_idx]).item() for selected_idx in selected)
            if max_sim > 0.8:
                continue
        selected.append(segments.index(seg))
    quotes = []
    for idx in selected:
        quotes.append({
            'text': segments[idx].text,
            'start': segments[idx].start,
            'end': segments[idx].end,
            'score': scores[idx]
        })
    return quotes

# ---------------------------
# NLP LEVEL 4: DISCOURSE INTEGRATION
# ---------------------------
def discourse_topic_modeling(chunks: List[Chunk], n_topics: int = 3, n_words: int = 5) -> Optional[Dict]:
    if not SKLEARN_AVAILABLE or len(chunks) < 3:
        return None
    texts = [c.text for c in chunks]
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    dtm = vectorizer.fit_transform(texts)
    if dtm.shape[1] < 2:
        return None
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(dtm)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-n_words-1:-1]]
        topics.append({"topic": topic_idx+1, "words": top_words})
    chunk_topics = lda.transform(dtm)
    topic_timeline = []
    for i, chunk in enumerate(chunks):
        dominant = chunk_topics[i].argmax() + 1
        topic_timeline.append({
            "chunk_index": i,
            "start": chunk.start,
            "end": chunk.end,
            "dominant_topic": dominant,
            "confidence": float(chunk_topics[i][dominant-1])
        })
    return {"topics": topics, "timeline": topic_timeline}

def discourse_coherence(chunks: List[Chunk]) -> Optional[List[float]]:
    if not SENTENCE_TRANSFORMERS_AVAILABLE or len(chunks) < 2:
        return None
    texts = [c.text for c in chunks]
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, convert_to_tensor=True)
    similarities = []
    for i in range(len(embeddings)-1):
        sim = util.cos_sim(embeddings[i], embeddings[i+1]).item()
        similarities.append(sim)
    return similarities

# ---------------------------
# NLP LEVEL 5: PRAGMATIC ANALYSIS
# ---------------------------
def pragmatic_answer_question(question: str, context: str) -> Optional[str]:
    if not TRANSFORMERS_AVAILABLE:
        return None
    try:
        qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")
        result = qa_pipeline(question=question, context=context[:20000])
        return result['answer']
    except Exception as e:
        st.warning(f"QA failed: {e}")
        return None

# ---------------------------
# SIDEBAR SETTINGS
# ---------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    auto_speed = st.checkbox("🚀 Auto speed optimization", value=True,
                             help="Automatically choose fastest settings based on video length.")

    # Initialize variables with defaults (will be overwritten in manual mode)
    model_choice = None
    temperature = 0.0
    chunk_size = 1200
    chunk_overlap = 200
    max_workers = get_cpu_count()
    chunk_timeout = 60
    include_timestamps_in_chunks = True
    custom_prompt = ""
    use_audio_fallback = True
    whisper_model = "tiny"
    use_faster = FASTER_WHISPER_AVAILABLE
    cookies_path = None
    split_minutes = 10
    max_minutes = 0
    chunk_strategy = "All chunks"
    sample_rate = 1
    max_chunks = 100
    use_browser_cookies = False
    browser_name = "chrome"

    if not auto_speed:
        model_options = {
            "phi3 (fastest)": "phi3",
            "gemma:2b (fast)": "gemma:2b",
            "mistral (medium)": "mistral",
            "llama3 (slow, best quality)": "llama3"
        }
        model_display = st.selectbox("Local LLM model", list(model_options.keys()), index=0)
        model_choice = model_options[model_display]
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        chunk_size = st.slider("Chunk size (chars)", 500, 2000, 1200, 100)
        chunk_overlap = st.slider("Overlap", 0, 500, 200, 50)
        max_workers = st.slider("Threads", 1, get_cpu_count(), min(4, get_cpu_count()))
        chunk_timeout = st.slider("Timeout per chunk (s)", 30, 300, 60, 10)
        st.markdown("---")
        st.header("🎤 Audio Transcription")
        use_audio_fallback = st.checkbox("Enable Whisper fallback", value=True)
        if use_audio_fallback:
            if not (WHISPER_AVAILABLE or FASTER_WHISPER_AVAILABLE):
                st.error("No Whisper backend installed.")
            use_faster = st.checkbox("Use faster-whisper (recommended)", value=FASTER_WHISPER_AVAILABLE)
            whisper_model = st.selectbox("Whisper model", ["tiny", "base", "small", "medium", "large"], index=0)

            # New: browser cookies option
            use_browser_cookies = st.checkbox("Use browser cookies (bypass 403)", value=False,
                                              help="Extract cookies from your browser (Chrome, Firefox, etc.)")
            if use_browser_cookies:
                browser_name = st.selectbox("Browser", ["chrome", "firefox", "edge", "opera", "brave"], index=0)

            # Manual cookies file (fallback)
            cookies_file = st.file_uploader("Or upload cookies file (.txt)", type=["txt"])

            split_minutes = st.slider("Split audio (minutes)", 1, 30, 10, 1)
            max_minutes = st.number_input("Process first N minutes (0 = no limit)", 0, value=120, step=10)
        st.markdown("---")
        st.header("📊 Summary Options")
        include_timestamps_in_chunks = st.checkbox("Include timestamps", value=True)
        custom_prompt = st.text_area("Custom prompt", height=100)
        st.markdown("---")
        st.header("⏱️ Long Video Optimization")
        with st.expander("Chunk reduction", expanded=False):
            chunk_strategy = st.radio("Strategy", ["All chunks", "Sample every N chunks", "First N chunks only"], index=0)
            if chunk_strategy == "Sample every N chunks":
                sample_rate = st.slider("Sample rate", 1, 10, 2, 1)
            elif chunk_strategy == "First N chunks only":
                max_chunks = st.number_input("Max chunks", 1, 1000, 50, step=10)
    else:
        # Auto mode: show minimal controls
        st.info("Auto mode will choose optimal settings.")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
        include_timestamps_in_chunks = st.checkbox("Include timestamps", value=True)
        custom_prompt = st.text_area("Custom prompt", height=100)
        st.info("💡 If you get a 403 error, disable auto mode and try browser cookies.")
        # Use default values for other settings

    st.markdown("---")
    st.header("📈 Evaluation")
    eval_option = st.radio("Reference for ROUGE/BERTScore", ["First 3000 chars of transcript", "Provide your own"], index=0)
    if eval_option == "Provide your own":
        reference_text = st.text_area("Paste reference summary")
    else:
        reference_text = None
    if BERTSCOPE_AVAILABLE:
        bert_model = st.selectbox("BERTScore model", ["roberta-large", "roberta-base", "distilbert-base-uncased"], index=0)
    else:
        bert_model = "roberta-large"
    verify = st.checkbox("🔍 Verification step", value=False)

    st.markdown("---")
    st.header("✨ NLP Feature Toggles")
    st.markdown("**Lexical**")
    enable_keywords = st.checkbox("Keywords", value=True)
    enable_wordfreq = st.checkbox("Word frequency", value=False)
    enable_postags = st.checkbox("POS distribution", value=False)

    st.markdown("**Syntactic**")
    enable_phrases = st.checkbox("Noun/Verb phrases", value=False)
    enable_dep = st.checkbox("Dependency relations", value=False)

    st.markdown("**Semantic**")
    enable_sentiment = st.checkbox("Sentiment timeline", value=True)
    enable_entities = st.checkbox("Named entities", value=True)
    enable_quotes = st.checkbox("Key quotes (improved)", value=True)

    st.markdown("**Discourse**")
    enable_topics = st.checkbox("Topic modeling (LDA)", value=False)
    enable_coherence = st.checkbox("Chunk coherence", value=False)

    st.markdown("**Pragmatic**")
    enable_qa = st.checkbox("Question answering", value=False)

# ---------------------------
# MAIN UI
# ---------------------------
url = st.text_input("🔗 YouTube URL", placeholder="https://youtu.be/...")
if url:
    video_id = extract_video_id(url)
    if video_id:
        st.video(f"https://youtu.be/{video_id}")
        st.caption("Confirm this is the correct video.")
    else:
        st.warning("Invalid YouTube URL.")

confirm = st.checkbox("✅ I confirm this is the correct video", value=False)

if st.button("🚀 Generate Summary & Transcription", type="primary", disabled=not confirm):
    if not url:
        st.error("Please enter a YouTube URL.")
        st.stop()

    video_id = extract_video_id(url)
    if not video_id:
        st.error("❌ Invalid YouTube URL.")
        st.stop()

    # Metadata
    duration, title = get_video_metadata(video_id)
    if duration:
        st.info(f"Video length: {duration // 60} min {duration % 60} sec")
    else:
        st.warning("Could not fetch video duration.")

    # Auto speed optimisation (overrides manual settings if enabled)
    if auto_speed:
        # Check captions quickly
        segments_test, _ = fetch_transcript_with_timestamps(video_id)
        captions_available = segments_test is not None

        model_choice = "phi3"
        chunk_size = 1500
        chunk_overlap = 200
        max_workers = get_cpu_count()
        chunk_timeout = 30
        include_timestamps_in_chunks = include_timestamps_in_chunks
        use_audio_fallback = True
        whisper_model = "tiny"
        use_faster = FASTER_WHISPER_AVAILABLE
        split_minutes = 15
        use_browser_cookies = False  # auto mode doesn't use browser cookies
        if duration:
            if duration > 7200:
                max_minutes = 30
            elif duration > 3600:
                max_minutes = 20
            else:
                max_minutes = 0
        else:
            max_minutes = 30

        if captions_available:
            use_audio_fallback = False
            st.info("✅ Captions found. Using them for fast processing.")
        else:
            st.info("🎤 No captions. Using tiny Whisper on limited portion.")

        if duration and duration > 10800:
            chunk_strategy = "Sample every N chunks"
            sample_rate = 3
            max_chunks = 50
        else:
            chunk_strategy = "All chunks"
            sample_rate = 1
            max_chunks = 100

    # Step 1: Get transcript
    with st.spinner("📥 Fetching transcript..."):
        segments, source = fetch_transcript_with_timestamps(video_id)

    # Step 2: Transcribe if needed
    if segments is None and use_audio_fallback:
        st.info("No captions found. Transcribing audio...")
        # Handle cookies file if uploaded
        if not auto_speed and 'cookies_file' in locals() and cookies_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                tmp.write(cookies_file.getvalue())
                cookies_path = tmp.name
        else:
            cookies_path = None

        # Call the improved function (no caching, with progress bar)
        segments, source = transcribe_audio_with_timestamps(
            video_id,
            whisper_model,
            cookies_path,
            max_minutes if max_minutes and max_minutes > 0 else None,
            split_minutes if split_minutes else 0,
            use_faster_whisper=use_faster,
            use_browser_cookies=use_browser_cookies,
            browser_name=browser_name
        )
        if segments is None:
            st.error(f"Transcription failed: {source}")
            st.stop()
        st.success("Audio transcription completed.")
    elif segments is None:
        st.error("No captions and audio fallback disabled.")
        st.stop()

    # Apply max_minutes to segments
    if max_minutes and max_minutes > 0:
        original = len(segments)
        segments = [s for s in segments if s.start <= max_minutes * 60]
        st.info(f"Limited to first {max_minutes} minutes: {len(segments)} segments kept (out of {original}).")
        if not segments:
            st.warning(f"No segments within {max_minutes} minutes. Increase limit.")
            st.stop()

    full_text = segments_to_text(segments)
    st.success(f"✅ Text obtained ({source}). Length: {len(full_text)} characters")
    st.info(f"Number of segments: {len(segments)}")

    # ----- Initialize variables for features (to avoid NameError in JSON export) -----
    keywords = []
    entities = {}
    quotes = []

    # ----- LEXICAL ANALYSIS -----
    st.markdown("---")
    st.subheader("📖 Lexical Analysis")
    if enable_keywords:
        with st.spinner("Extracting keywords..."):
            keywords = lexical_analysis_keywords(full_text)
            if keywords:
                st.markdown("**Keywords:** " + ", ".join(keywords))
    if enable_wordfreq:
        with st.spinner("Computing word frequencies..."):
            wordfreq = lexical_analysis_wordfreq(full_text)
            if wordfreq:
                st.markdown("**Top words:** " + ", ".join([f"{w} ({c})" for w, c in wordfreq]))
    if enable_postags and SPACY_AVAILABLE:
        with st.spinner("Analyzing POS tags..."):
            postags = lexical_analysis_postags(full_text)
            if postags:
                st.markdown("**POS distribution:** " + ", ".join([f"{tag}:{count}" for tag, count in postags.items()]))

    # ----- SYNTACTIC ANALYSIS -----
    st.markdown("---")
    st.subheader("🔧 Syntactic Analysis")
    if enable_phrases and SPACY_AVAILABLE:
        with st.spinner("Extracting phrases..."):
            phrases = syntactic_analysis_phrases(full_text)
            if phrases:
                st.markdown("**Noun phrases (sample):** " + ", ".join(phrases.get("noun_phrases", [])))
                st.markdown("**Verb phrases (sample):** " + ", ".join(phrases.get("verb_phrases", [])))
    if enable_dep and SPACY_AVAILABLE:
        with st.spinner("Analyzing dependency relations..."):
            deps = syntactic_analysis_dependencies(full_text)
            if deps:
                st.markdown("**Top dependency relations:** " + ", ".join([f"{rel}:{count}" for rel, count in deps.items()]))

    # ----- SEMANTIC ANALYSIS -----
    st.markdown("---")
    st.subheader("🧠 Semantic Analysis")
    if enable_sentiment and TEXTBLOB_AVAILABLE and PLOTLY_AVAILABLE:
        with st.spinner("Computing sentiment timeline..."):
            timeline = semantic_sentiment_timeline(segments)
            if timeline:
                df = pd.DataFrame(timeline)
                fig = px.line(df, x='start', y='polarity', title='Sentiment Polarity Over Time')
                st.plotly_chart(fig, use_container_width=True)
    if enable_entities and SPACY_AVAILABLE:
        with st.spinner("Extracting named entities..."):
            entities = semantic_extract_entities(full_text)
            if entities:
                st.markdown("**Named Entities**")
                for label, values in entities.items():
                    st.markdown(f"*{label}:* {', '.join(values[:10])}")
    if enable_quotes:
        with st.spinner("Extracting key quotes (semantic)..."):
            quotes = semantic_key_quotes(segments)
            if quotes:
                st.markdown("**Key Quotes**")
                for q in quotes:
                    st.markdown(f"- *{q['text']}*  \n  — at {format_timestamp(q['start'])} (score: {q['score']:.2f})")

    # ----- DISCOURSE INTEGRATION -----
    st.markdown("---")
    st.subheader("🔗 Discourse Integration")
    # Need chunks for discourse
    chunks = chunk_segments(segments, chunk_size, chunk_overlap)
    if enable_topics and SKLEARN_AVAILABLE and len(chunks) >= 3:
        with st.spinner("Performing topic modeling..."):
            topics_data = discourse_topic_modeling(chunks)
            if topics_data:
                st.markdown("**Main Topics**")
                for topic in topics_data["topics"]:
                    st.markdown(f"Topic {topic['topic']}: " + ", ".join(topic["words"]))
                if PLOTLY_AVAILABLE:
                    df_topics = pd.DataFrame(topics_data["timeline"])
                    fig = px.scatter(df_topics, x='start', y='dominant_topic', color='confidence',
                                     title='Topic Evolution Over Time',
                                     labels={'start': 'Time (s)', 'dominant_topic': 'Topic'})
                    st.plotly_chart(fig, use_container_width=True)
    if enable_coherence and SENTENCE_TRANSFORMERS_AVAILABLE and len(chunks) >= 2:
        with st.spinner("Computing chunk coherence..."):
            coh = discourse_coherence(chunks)
            if coh and PLOTLY_AVAILABLE:
                df_coh = pd.DataFrame({"chunk": list(range(1, len(coh)+1)), "similarity": coh})
                fig = px.line(df_coh, x='chunk', y='similarity', title='Coherence Between Consecutive Chunks')
                st.plotly_chart(fig, use_container_width=True)

    # ----- PRAGMATIC ANALYSIS -----
    if enable_qa and TRANSFORMERS_AVAILABLE:
        st.markdown("---")
        st.subheader("❓ Pragmatic Analysis (Question Answering)")
        question = st.text_input("Ask a question about the video content:")
        if question:
            with st.spinner("Searching for answer..."):
                answer = pragmatic_answer_question(question, full_text)
                if answer:
                    st.success(f"**Answer:** {answer}")
                else:
                    st.warning("No answer found.")

    # ----- Proceed with summarization -----
    st.info(f"📄 Split into {len(chunks)} chunks")
    chunks = filter_chunks_by_strategy(chunks, chunk_strategy, sample_rate, max_chunks)

    # LLM
    try:
        llm = Ollama(model=model_choice, temperature=temperature)
        llm.invoke("Say 'OK'")
    except Exception as e:
        if "404" in str(e):
            st.error(f"❌ Model '{model_choice}' not found. Run: `ollama pull {model_choice}`")
        else:
            st.error(f"❌ Ollama error: {e}")
        st.stop()

    parser = StrOutputParser()
    base_prompt = PromptTemplate.from_template("Summarize: {text}\nSummary:")
    summarization_chain = base_prompt | llm | parser

    with st.spinner(f"🧠 Summarizing {len(chunks)} chunks..."):
        start_time = time.time()
        chunk_summaries = [None] * len(chunks)
        errors = []
        progress = st.progress(0, text="Summarizing...")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(summarize_chunk, chunks[i], summarization_chain, include_timestamps_in_chunks, chunk_timeout): i
                for i in range(len(chunks))
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    res = future.result(timeout=chunk_timeout)
                    if res:
                        chunk_summaries[idx] = res
                    else:
                        errors.append(f"Chunk {idx+1}: failed")
                except TimeoutError:
                    errors.append(f"Chunk {idx+1}: timeout")
                except Exception as e:
                    errors.append(f"Chunk {idx+1}: {e}")
                progress.progress((sum(1 for s in chunk_summaries if s is not None) + len(errors)) / len(chunks))

        successful = [s for s in chunk_summaries if s is not None]
        if not successful:
            st.error("All chunks failed.")
            st.stop()
        if errors:
            st.warning(f"⚠️ {len(errors)} chunks failed. Using {len(successful)} successful.")
            with st.expander("Show errors"):
                for err in errors[:5]:
                    st.write(err)

        final = final_summary(successful, llm, custom_prompt if custom_prompt else None)
        elapsed = time.time() - start_time
        progress.empty()

    st.subheader("📌 Final Summary")
    st.write(final)
    st.caption(f"Generated in {elapsed:.2f}s with {model_choice} (temp={temperature})")

    st.subheader("📜 Chunk Summaries")
    for i, s in enumerate(chunk_summaries):
        if s:
            st.markdown(f"**Chunk {i+1}:** {s}")

    st.subheader("📝 Full Transcription")
    transcription_text = format_transcription(segments)
    st.text_area("Transcription", transcription_text, height=300)

    # Evaluation
    st.markdown("---")
    st.subheader("📈 Evaluation")
    if eval_option == "First 3000 chars of transcript":
        reference = full_text[:3000]
    else:
        reference = reference_text
    if reference:
        rouge = evaluate_rouge(reference, final)
        if rouge:
            st.metric("ROUGE-L", f"{rouge:.4f}")
        if BERTSCOPE_AVAILABLE:
            with st.spinner("BERTScore..."):
                bert = evaluate_bertscore(reference, final, bert_model)
                if bert:
                    st.metric("BERTScore F1", f"{bert:.4f}")
    else:
        st.info("No reference provided.")

    # Verification
    if verify:
        with st.spinner("Verifying..."):
            try:
                verified = verify_summary(full_text, final, llm)
                st.subheader("✅ Verified Summary")
                st.write(verified)
            except Exception as e:
                st.error(f"Verification failed: {e}")

    # Export
    st.subheader("📥 Export")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.download_button("Download Summary (TXT)", final, file_name=f"{video_id}_summary.txt")
    with col2:
        chunk_txt = "\n\n".join([s for s in chunk_summaries if s])
        st.download_button("Download Chunks (TXT)", chunk_txt, file_name=f"{video_id}_chunks.txt")
    with col3:
        st.download_button("Download Transcript (TXT)", transcription_text, file_name=f"{video_id}_transcript.txt")
    with col4:
        data = {
            "video_id": video_id,
            "title": title,
            "source": source,
            "final_summary": final,
            "chunk_summaries": [s for s in chunk_summaries if s],
            "transcription": transcription_text,
            "keywords": keywords if enable_keywords else [],
            "entities": entities if enable_entities and SPACY_AVAILABLE else {},
            "quotes": quotes if enable_quotes else []
        }
        st.download_button("Download JSON", json.dumps(data, indent=2), file_name=f"{video_id}.json", mime="application/json")

    # Cleanup
    if 'cookies_path' in locals() and cookies_path and os.path.exists(cookies_path):
        try:
            os.unlink(cookies_path)
        except:
            pass
