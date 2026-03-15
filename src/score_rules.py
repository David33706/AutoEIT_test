"""
Rule-based EIT scoring using the Ortega (2000) meaning-based rubric.
Scores 0-4 based on word overlap, content preservation, and meaning.
"""

import re
import unicodedata
import Levenshtein

# Spanish function words
FUNCTION_WORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "en", "a", "al", "con", "por", "para", "se",
    "que", "y", "o", "pero", "no", "muy", "mas", "me", "mi",
    "su", "sus", "lo", "le", "les", "es", "son", "fue", "ha",
    "he", "yo", "el", "ella", "nosotros", "usted", "ustedes",
    "tan", "todo", "toda", "todos", "todas", "cada", "este",
    "esta", "esto", "eso", "como", "cuando", "si", "ya",
}

# Known false-friend pairs that should NOT fuzzy-match
FALSE_FRIENDS = {
    ("ducha", "lucha"), ("lucha", "ducha"),
    ("casa", "cosa"), ("cosa", "casa"),
    ("pero", "perro"), ("perro", "pero"),
    ("carro", "caro"), ("caro", "carro"),
}


def strip_accents(text):
    """Remove accent marks. peliculas == películas"""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def clean(text):
    """Normalize text for comparison."""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\bx+\b', '', text)
    text = re.sub(r'\bm+h+\b', '', text)
    text = re.sub(r'\buf\b', '', text)
    text = re.sub(r'\bmeh\b', '', text)
    text = re.sub(r'\b\w+\-\s', '', text)
    text = re.sub(r'\b\w\-', '', text)
    text = re.sub(r'[¿?¡!.,;:\'"…/]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = strip_accents(text)
    return text


def fuzzy_word_match(word1, word2, threshold=0.75):
    """Check if two words are similar enough, excluding known false friends."""
    if word1 == word2:
        return True
    if not word1 or not word2:
        return False
    if (word1, word2) in FALSE_FRIENDS:
        return False
    sim = 1 - Levenshtein.distance(word1, word2) / max(len(word1), len(word2))
    return sim >= threshold


def get_content_words(text):
    """Extract content words (non-function words)."""
    words = clean(text).split()
    return [w for w in words if w not in FUNCTION_WORDS and len(w) > 1]


def get_all_words(text):
    """Get all words from cleaned text."""
    return clean(text).split()


def word_overlap(target_words, response_words):
    """Fraction of target words appearing in response (fuzzy)."""
    if not target_words:
        return 0
    matched = sum(1 for w in target_words
                  if any(fuzzy_word_match(w, rw) for rw in response_words))
    return matched / len(target_words)


def content_overlap(target, response):
    """Content word overlap ratio (fuzzy)."""
    t_content = get_content_words(target)
    r_content = get_content_words(response)
    if not t_content:
        return 0
    matched = sum(1 for w in t_content
                  if any(fuzzy_word_match(w, rw) for rw in r_content))
    return matched / len(t_content)


def levenshtein_sim(target, response):
    """Normalized Levenshtein similarity on cleaned text."""
    a = clean(target)
    b = clean(response)
    if not a or not b:
        return 0
    return 1 - Levenshtein.distance(a, b) / max(len(a), len(b))


def is_empty_response(response):
    """Check if response is essentially empty."""
    cleaned = clean(response)
    return len(cleaned) == 0 or cleaned in ['', ' ']


def score_eit(target, response):
    """
    Score an EIT response on 0-4 scale.

    4 = exact repetition (form + meaning correct)
    3 = meaning preserved, minor grammar changes okay
    2 = more than half of ideas, meaning close but inexact
    1 = about half of ideas, lots missing
    0 = silence, garbled, or minimal repetition
    """
    if not response or is_empty_response(response):
        return 0

    t_words = get_all_words(target)
    r_words = get_all_words(response)
    t_content = get_content_words(target)
    r_content = get_content_words(response)

    lev_sim = levenshtein_sim(target, response)
    content_ov = content_overlap(target, response)
    word_ov = word_overlap(t_words, r_words)
    r_content_count = sum(1 for w in t_content
                          if any(fuzzy_word_match(w, rw) for rw in r_content))
    r_word_count = len(r_words)

    # === Score 4: Exact or near-exact ===
    if content_ov >= 1.0 and lev_sim >= 0.90:
        return 4

    # === Score 3: Meaning preserved ===
    if content_ov >= 0.80 and lev_sim >= 0.60:
        return 3
    if content_ov >= 0.85 and word_ov >= 0.60:
        return 3
    if content_ov >= 0.80 and r_word_count >= len(t_words) * 0.5:
        return 3

    # === Score 2: More than half of ideas ===
    if content_ov >= 0.50 and lev_sim >= 0.40:
        return 2
    if content_ov >= 0.60 and r_word_count >= 3:
        return 2
    if word_ov >= 0.50 and lev_sim >= 0.50 and r_word_count >= 4:
        return 2

    # === Score 1: About half, lots missing ===
    if content_ov >= 0.25 and r_word_count >= 3:
        return 1
    if r_content_count >= 2 and r_word_count >= 3:
        return 1
    if word_ov >= 0.30 and r_word_count >= 3:
        return 1

    # === Score 0 ===
    return 0


if __name__ == "__main__":
    test_cases = [
        ("Quiero cortarme el pelo", "Quiero cortarme el pelo", 4),
        ("El carro lo tiene Pedro", "El carro tiene Pedro", 3),
        ("Dudo que sepa manejar muy bien", "Dudo que sepa manajar bien", 3),
        ("El chico con el que yo salgo es español", "El chico con yo salgo ...um.. está bien", 2),
        ("Después de cenar me fui a dormir tranquilo", "Despues de cenar fue en- tranquilo", 1),
        ("Las calles de esta ciudad son muy anchas", "Las calles..es-[gibberish]...", 0),
        ("Quiero cortarme el pelo", "Manaña", 0),
        ("El se ducha cada mañana", "El se lucha cada mañana", 2),  # meaning change!
    ]

    print(f"{'Score':>5} | {'Exp':>3} | Target → Response")
    print("-" * 80)
    for target, response, expected in test_cases:
        score = score_eit(target, response)
        match = "✓" if score == expected else "✗"
        print(f"  {score}  {match} | {expected:>3} | {target} → {response}")