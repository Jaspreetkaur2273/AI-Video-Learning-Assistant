"""
llm_service.py — AI/NLP pipeline for generating learning content from transcripts.
Uses Hugging Face transformers for summarization and text generation.
"""

import re
import json
import textwrap
import torch
from transformers import pipeline

import time
import concurrent.futures

# ---------------------------------------------------------------------------
# Model Initialization (Eager Loading for instant response)
# ---------------------------------------------------------------------------

def _get_device_str():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

print(f"[llm_service] Initializing AI models on device: {_get_device_str()}...")

try:
    start_time = time.time()
    device = _get_device_str()
    # Use FP16 for speed if on GPU
    dtype = torch.float16 if device in ["mps", "cuda"] else torch.float32
    
    _summarizer_pipe = pipeline(
        "summarization",
        model="facebook/bart-large-cnn", # Stronger summarizer
        device=device,
        model_kwargs={"torch_dtype": dtype}
    )

    _generator_pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base", # Better generator
        device=device,
        model_kwargs={"torch_dtype": dtype},
        max_new_tokens=1024,
    )
    # Optimized settings for speed and stability
    _gen_kwargs = {
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3,
        "early_stopping": True,
        "batch_size": 8, # Increased batch size for faster MPS inference
    }

    print(f"[llm_service] Optimized models loaded in {time.time() - start_time:.2f}s")
except Exception as e:
    print(f"[llm_service] Error loading optimized models: {e}")
    _summarizer_pipe = pipeline("summarization", model="facebook/bart-large-cnn")
    _generator_pipe = pipeline("text2text-generation", model="google/flan-t5-base", max_new_tokens=512)
    _gen_kwargs = {}

# ---------------------------------------------------------------------------
# Helpers: chunking and timestamps
# ---------------------------------------------------------------------------

def _chunk_text(text: str, max_chars: int = 2000) -> list[str]:
    """Split text into chunks of roughly max_chars characters at sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) > max_chars and current:
            chunks.append(current.strip())
            current = ""
        current += " " + sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks if chunks else [text[:max_chars]]

def _get_timestamp_str(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS string."""
    s = int(seconds)
    hours = s // 3600
    minutes = (s % 3600) // 60
    seconds = s % 60
    if hours > 0:
        return f"[{hours:02}:{minutes:02}:{seconds:02}]"
    return f"[{minutes:02}:{seconds:02}]"

def _get_sampled_chunks_with_times(segments: list, max_chars: int = 2000, num_samples: int = 4) -> list:
    """Split segments into chunks while tracking the start time of each chunk."""
    if not segments:
        return []
        
    all_chunks = []
    current_text = ""
    current_start = segments[0]["start"]
    
    for seg in segments:
        if len(current_text) + len(seg["text"]) > max_chars and current_text:
            all_chunks.append({
                "text": current_text.strip(),
                "start": current_start,
                "timestamp": _get_timestamp_str(current_start)
            })
            current_text = ""
            current_start = seg["start"]
        current_text += " " + seg["text"]
        
    if current_text.strip():
        all_chunks.append({
            "text": current_text.strip(),
            "start": current_start,
            "timestamp": _get_timestamp_str(current_start)
        })
        
    if len(all_chunks) <= num_samples:
        return all_chunks
        
    indices = [0]
    if num_samples > 1:
        step = (len(all_chunks) - 1) / (num_samples - 1)
        for i in range(1, num_samples - 1):
            idx = int(i * step)
            if idx not in indices:
                indices.append(idx)
        indices.append(len(all_chunks) - 1)
        
    return [all_chunks[i] for i in sorted(list(set(indices)))]

# ---------------------------------------------------------------------------
# Public AI Task Components
# ---------------------------------------------------------------------------

def generate_summary(transcript: str) -> str:
    """Generate a highly detailed summary rapidly by concatenating chunk completions without slow single-pass refinement."""
    t0 = time.time()
    chunks = _chunk_text(transcript, max_chars=3000)
    if not chunks: return ""
    
    # Step 1: Detailed summaries of chunks
    with torch.inference_mode():
        # Using high max_length to extract highly detailed text
        chunk_results = _summarizer_pipe(chunks, max_length=250, min_length=80, do_sample=False, truncation=True, **_gen_kwargs)
    
    intermediate_summaries = [r["summary_text"] for r in chunk_results if "summary_text" in r]
    
    # Skip the slow generation step entirely for massive speedup and detailed paragraph output
    final_summary = "\n\n".join(intermediate_summaries)

    print(f"[llm_service] High-Speed Summary generated in {time.time() - t0:.2f}s (chunks: {len(chunks)})")
    return final_summary

def generate_category(transcript: str) -> str:
    """Identify the primary subject category of the video."""
    t0 = time.time()
    # Use just the first part for categorization
    text = transcript[:2000]
    prompt = f"Identify the primary educational or topical category for this content (e.g. Technology, Science, History, Health, Business, Personal Development, Cooking, etc.). Respond with ONE OR TWO WORDS ONLY.\n\nContent: {text}\n\nCategory:"
    
    try:
        with torch.inference_mode():
            result = _generator_pipe(prompt, max_new_tokens=10, truncation=True)
        category = result[0]["generated_text"].strip().title().replace(".", "")
        print(f"[llm_service] Category identified: {category} in {time.time() - t0:.2f}s")
        return category if category else "General"
    except:
        return "General"

def generate_key_points(segments: list) -> list[str]:
    """Extract key actionable points from the video transcript."""
    t0 = time.time()
    chunks = _get_sampled_chunks_with_times(segments, max_chars=1500, num_samples=3)
    if not chunks: return []

    prompts = [
        "Extract 3 main key facts from this text. Make them short bullet points.\n\n"
        f"Text: {c['text']}\n\nFacts:"
        for c in chunks
    ]
    with torch.inference_mode():
        results = _generator_pipe(prompts, max_new_tokens=150, truncation=True, **_gen_kwargs)
        
    all_points = []
    for r in results:
        text = r["generated_text"].strip()
        points = [p.strip().lstrip('*-•1234567890. ') for p in text.split("\n") if p.strip()]
        all_points.extend([p for p in points if len(p) > 10])
        
    if not all_points:
        return ["Central concepts discussed in the video", "Important details and examples", "Closing arguments or conclusions"]
        
    print(f"[llm_service] Key points generated in {time.time() - t0:.2f}s")
    return all_points[:10]

def generate_discovery_questions(summary: str) -> list[str]:
    """Generate follow-up research questions based on the summary."""
    t0 = time.time()
    prompt = (
        "Based on this summary, suggest 3 advanced follow-up questions for further learning. "
        "Keep them thought-provoking. Respond as a JSON list of strings.\n\n"
        f"Summary: {summary}\n\nQuestions:"
    )
    try:
        with torch.inference_mode():
            result = _generator_pipe(prompt, max_new_tokens=150, truncation=True)
        raw = result[0]["generated_text"].strip()
        # Fallback parsing if LLM doesn't output clean JSON
        questions = re.findall(r'"([^"]+)"', raw)
        if not questions:
            questions = [q.strip() for q in raw.split("\n") if q.strip()][:3]
        print(f"[llm_service] Discovery questions generated in {time.time() - t0:.2f}s")
        return questions[:3]
    except:
        return ["What are the long-term implications of this topic?", "How does this connect to other related fields?", "What are the common misconceptions about this?"]

def generate_notes(segments: list) -> str:
    """Generate detailed study notes without timestamps for faster, cleaner output."""
    t0 = time.time()
    # Speed optimization: Use fewer chunks, process them efficiently
    chunks = _get_sampled_chunks_with_times(segments, max_chars=2500, num_samples=4)
    if not chunks: return ""

    prompts = [
        "Comprehensively summarize the following text into highly detailed bullet points with specific facts:\n\n"
        f"Text:\n{c['text']}\n\nBullet Notes:"
        for c in chunks
    ]
    
    with torch.inference_mode():
        results = _generator_pipe(prompts, max_new_tokens=400, truncation=True, **_gen_kwargs)
    
    note_chunks = []
    for r in results:
        text = r["generated_text"].strip()
        note_chunks.append(text)
        
    notes = "\n\n".join(note_chunks)
    print(f"[llm_service] Detailed Notes generated in {time.time() - t0:.2f}s")
    return notes

def generate_quiz(segments: list) -> list[dict]:
    """Generate a quiz covering the entire video with a more reliable prompt."""
    t0 = time.time()
    chunks = _get_sampled_chunks_with_times(segments, max_chars=1800, num_samples=3)
    if not chunks: return []
    
    prompts = [
        "Create a specific multiple choice question based on the facts in this text. Provide 4 distinct options.\n\n"
        f"Text: {c['text']}\n\n"
        "Format output like this:\n"
        "Question: [Specific question based on text]\n"
        "A) [Plausible answer 1]\n"
        "B) [Plausible answer 2]\n"
        "C) [Plausible answer 3]\n"
        "D) [Plausible answer 4]\n"
        "Answer: A"
        for c in chunks
    ]
    
    with torch.inference_mode():
        results = _generator_pipe(prompts, max_new_tokens=250, truncation=True, **_gen_kwargs)
    
    full_quiz_raw = "\n\n".join([r["generated_text"] for r in results])
    quiz = _parse_quiz(full_quiz_raw, " ".join([c["text"] for c in chunks]))
    
    print(f"[llm_service] Quiz generated in {time.time() - t0:.2f}s")
    return quiz

def _parse_quiz(raw: str, transcript: str) -> list[dict]:
    """Parse the LLM quiz output into structured JSON with flexible regex."""
    questions = []
    # Split by keywords that likely start a new question block
    blocks = re.split(r'(?i)Question:?\s*', raw)
    
    for block in blocks:
        if not block.strip(): continue
        
        # Extract question text (up to the first option)
        q_text_match = re.match(r'(.*?)(?=[A-D][\).:]\s*)', block.strip(), re.DOTALL | re.IGNORECASE)
        q_text = q_text_match.group(1).strip() if q_text_match else block.strip().split("\n")[0]
        
        # Extract options
        opts = re.findall(r'([A-D])[\).:]\s*(.+)', block, re.IGNORECASE)
        options = [o[1].strip() for o in opts]
        
        # Extract answer
        ans_match = re.search(r'Answer:\s*([A-D])', block, re.IGNORECASE)
        correct = 0
        if ans_match:
            correct = ord(ans_match.group(1).upper()) - ord('A')
        
        if q_text and len(options) >= 2:
            questions.append({
                "question": q_text,
                "options": options[:4],
                "correct": min(max(0, correct), len(options) - 1),
            })
            
    if not questions:
        # Emergency fallback question
        questions.append({
            "question": "What is the primary objective of this video tutorial?",
            "options": ["To introduce the core concepts of the topic", "To provide a detailed historical background", "To review viewer comments", "To compare unrelated software"],
            "correct": 0
        })
        
    return questions[:10]

# ---------------------------------------------------------------------------
# Combined Unified Pipeline (Parallel)
# ---------------------------------------------------------------------------

def process_transcript(transcript_text: str, segments: list, target_language: str = "en") -> dict:
    """Run the optimized parallel processing pipeline with maximum concurrency."""
    overall_start = time.time()
    print(f"[llm_service] Starting high-speed parallel analysis...")
    
    is_translation_needed = target_language and target_language.lower() != "en"
    
    def translate_if_needed(text):
        if not is_translation_needed or not text or len(text) < 3: return text
        from deep_translator import GoogleTranslator
        try:
            translator = GoogleTranslator(source='auto', target=target_language.lower())
            if len(text) <= 4500: return translator.translate(text)
            chunks = [text[i:i+4500] for i in range(0, len(text), 4500)]
            return " ".join([translator.translate(c) for c in chunks])
        except: return text

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Step 1: Core generations
        f_summary = executor.submit(generate_summary, transcript_text)
        f_cat = executor.submit(generate_category, transcript_text)
        f_kp = executor.submit(generate_key_points, segments)
        f_notes = executor.submit(generate_notes, segments)
        f_quiz = executor.submit(generate_quiz, segments)

        # Step 2: Extract results and trigger downstream tasks
        category = f_cat.result()
        key_points = f_kp.result()
        summary = f_summary.result()
        notes = f_notes.result()
        quiz = f_quiz.result()
        
        # Step 3: Dependent tasks
        f_discovery = executor.submit(generate_discovery_questions, summary)
        
        # Step 4: Parallel Translation
        if is_translation_needed:
            t_summary = executor.submit(translate_if_needed, summary)
            t_notes = executor.submit(translate_if_needed, notes)
            t_kp = executor.submit(lambda: [translate_if_needed(kp) for kp in key_points])
            
            summary = t_summary.result()
            notes = t_notes.result()
            key_points = t_kp.result()
            for q in quiz:
                q["question"] = translate_if_needed(q["question"])
                q["options"] = [translate_if_needed(opt) for opt in q["options"]]
        
        discovery = f_discovery.result()
        if is_translation_needed:
            discovery = [translate_if_needed(d) for d in discovery]

    total_time = time.time() - overall_start
    print(f"[llm_service] High-speed pipeline complete in {total_time:.2f}s")

    return {
        "summary": summary,
        "category": category,
        "key_points": key_points,
        "notes": notes,
        "quiz": quiz,
        "discovery": discovery
    }
