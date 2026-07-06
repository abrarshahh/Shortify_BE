from typing import List, Dict, Any

def chunk_words(
    words: List[Dict[str, Any]],
    strategy: str,
    max_words: int = 6,
    max_duration: float = 2.5
) -> List[Dict[str, Any]]:
    """
    Groups a flat list of Whisper words into caption chunks based on the chosen strategy.
    
    Each word is a dict: {"word": str, "start": float, "end": float}
    Returns a list of caption chunks:
      [
        {
          "start": float,
          "end": float,
          "text": str,
          "words": [{"word": str, "start": float, "end": float}, ...]
        },
        ...
      ]
    """
    if not words:
        return []

    # Clean and standardize words
    clean_words = []
    for w in words:
        word_text = w.get("word", "").strip()
        # Keep word entries valid
        clean_words.append({
            "word": word_text,
            "start": round(float(w["start"]), 2),
            "end": round(float(w["end"]), 2)
        })

    chunks = []
    if strategy == "single_word":
        for w in clean_words:
            chunks.append([w])
    elif strategy == "duration_capped":
        current_chunk = []
        chunk_start = None
        for w in clean_words:
            if not current_chunk:
                current_chunk.append(w)
                chunk_start = w["start"]
            else:
                dur = w["end"] - chunk_start
                if len(current_chunk) >= max_words or dur > max_duration:
                    chunks.append(current_chunk)
                    current_chunk = [w]
                    chunk_start = w["start"]
                else:
                    current_chunk.append(w)
        if current_chunk:
            chunks.append(current_chunk)
    elif strategy == "punctuation_aware":
        current_chunk = []
        chunk_start = None
        for w in clean_words:
            if not current_chunk:
                current_chunk.append(w)
                chunk_start = w["start"]
            else:
                current_chunk.append(w)
                
            # Check for sentence and clause boundaries
            word_text = w["word"]
            is_sentence_end = word_text.endswith((".", "?", "!"))
            is_clause_end = word_text.endswith((",", ";"))
            
            dur = w["end"] - chunk_start
            
            # Break decisions
            if is_sentence_end:
                chunks.append(current_chunk)
                current_chunk = []
                chunk_start = None
            elif is_clause_end and len(current_chunk) >= 3:
                chunks.append(current_chunk)
                current_chunk = []
                chunk_start = None
            elif len(current_chunk) >= max_words or dur > max_duration:
                chunks.append(current_chunk)
                current_chunk = []
                chunk_start = None
        if current_chunk:
            chunks.append(current_chunk)
    else:  # fallback to fixed_word_count
        for i in range(0, len(clean_words), max_words):
            chunks.append(clean_words[i : i + max_words])

    # Convert word groups to caption chunk format
    formatted_chunks = []
    for group in chunks:
        if not group:
            continue
        formatted_chunks.append({
            "start": round(group[0]["start"], 2),
            "end": round(group[-1]["end"], 2),
            "text": " ".join(w["word"] for w in group),
            "words": group
        })
    return formatted_chunks
