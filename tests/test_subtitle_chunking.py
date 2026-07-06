import pytest
from backend_ai.effects.caption_chunking import chunk_words
from backend_ai.agents.subtitle_agent import SubtitleAgent

def test_chunk_words_single_word():
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5},
        {"word": "world", "start": 0.5, "end": 1.0},
        {"word": "how", "start": 1.0, "end": 1.5},
        {"word": "are", "start": 1.5, "end": 2.0},
        {"word": "you", "start": 2.0, "end": 2.5}
    ]
    chunks = chunk_words(words, strategy="single_word")
    assert len(chunks) == 5
    assert chunks[0]["text"] == "Hello"
    assert chunks[4]["text"] == "you"
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["end"] == 0.5

def test_chunk_words_fixed_word_count():
    words = [
        {"word": "One", "start": 0.0, "end": 0.5},
        {"word": "two", "start": 0.5, "end": 1.0},
        {"word": "three", "start": 1.0, "end": 1.5},
        {"word": "four", "start": 1.5, "end": 2.0},
        {"word": "five", "start": 2.0, "end": 2.5}
    ]
    chunks = chunk_words(words, strategy="fixed_word_count", max_words=3)
    assert len(chunks) == 2
    assert chunks[0]["text"] == "One two three"
    assert chunks[1]["text"] == "four five"
    assert chunks[0]["start"] == 0.0
    assert chunks[0]["end"] == 1.5
    assert chunks[1]["start"] == 1.5
    assert chunks[1]["end"] == 2.5

def test_chunk_words_duration_capped():
    words = [
        {"word": "A", "start": 0.0, "end": 0.5},
        {"word": "fast", "start": 0.5, "end": 1.0},
        {"word": "sentence", "start": 1.0, "end": 3.0}, # duration = 3.0 - 0.0 = 3.0 > 2.5
        {"word": "here", "start": 3.0, "end": 3.5}
    ]
    chunks = chunk_words(words, strategy="duration_capped", max_words=6, max_duration=2.5)
    assert len(chunks) == 2
    assert chunks[0]["text"] == "A fast"
    assert chunks[1]["text"] == "sentence here"

def test_chunk_words_punctuation_aware():
    # Sentence boundary end
    words = [
        {"word": "Hello.", "start": 0.0, "end": 0.5},
        {"word": "How", "start": 0.5, "end": 1.0},
        {"word": "are", "start": 1.0, "end": 1.5},
        {"word": "you?", "start": 1.5, "end": 2.0},
        {"word": "I", "start": 2.0, "end": 2.5},
        {"word": "am", "start": 2.5, "end": 3.0},
        {"word": "fine,", "start": 3.0, "end": 3.5}, # Clause separator comma, size = 3
        {"word": "thank", "start": 3.5, "end": 4.0},
        {"word": "you.", "start": 4.0, "end": 4.5}
    ]
    chunks = chunk_words(words, strategy="punctuation_aware", max_words=6, max_duration=4.0)
    # Expected chunks:
    # 1. "Hello." (breaks on sentence-ending punctuation)
    # 2. "How are you?" (breaks on sentence-ending punctuation)
    # 3. "I am fine," (breaks on comma because size = 3 >= 3)
    # 4. "thank you." (breaks on sentence-ending punctuation)
    assert len(chunks) == 4
    assert chunks[0]["text"] == "Hello."
    assert chunks[1]["text"] == "How are you?"
    assert chunks[2]["text"] == "I am fine,"
    assert chunks[3]["text"] == "thank you."

def test_subtitle_agent_build_captions():
    agent = SubtitleAgent(caption_style="hormozi")
    segments = [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "Hello world",
            "words": [
                {"word": "Hello", "start": 0.0, "end": 1.0},
                {"word": "world", "start": 1.0, "end": 2.0}
            ]
        }
    ]
    # hormozi style should yield single_word chunking strategy
    captions = agent._build_captions(segments, style_name="hormozi")
    assert len(captions) == 2
    assert captions[0]["text"] == "Hello"
    assert captions[1]["text"] == "world"

    # minimal style should yield punctuation_aware
    captions_min = agent._build_captions(segments, style_name="minimal")
    assert len(captions_min) == 1
    assert captions_min[0]["text"] == "Hello world"
