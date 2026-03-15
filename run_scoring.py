"""
Rule-based EIT scoring using the Ortega (2000) meaning-based rubric.
Scores 0-4 based on word overlap, content preservation, and meaning.
"""

import re
import Levenshtein

# Spanish function words (articles, prepositions, pronouns, conjunctions)
FUNCTION_WORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "en", "a", "al", "con", "por", "para", "se",
    "que", "y", "o", "pero", "no", "muy", "más", "me", "mi",
    "su", "sus", "lo", "le", "les", "es", "son", "fue", "ha",
    "he", "yo", "él", "ella", "nosotros", "usted", "ustedes",
    "tan", "todo", "toda", "todos", "todas", "cada", "este",
    "esta", "esto", "eso", "como", "cuando", "si", "ya",
}


def clean(text):
    """Normalize text for comparison."""
    text = text.lower()
    # Remove annotations like [pause], [gibberish], [cough], (xxx?), etc.
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    # Remove disfluency markers: xxx, xx, x, mhh, uf
    text = re.sub(r'\bx+\b', '', text)
    text = re.sub(r'\bm+h+\b', '', text)
    text = re.sub(r'\buf\b', '', text)
    text = re.sub(r'\bmeh\b', '', text)
    # Remove false starts (word fragments ending with -)
    text = re.sub(r'\b\w+\-\s', '', text)
    # Remove leading fragments like "e-"
    text = re.sub(r'\b\w\-', '', text)
    # Remove punctuation
    text = re.sub(r'[¿?¡!.,;:\'"…]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def fuzzy_word_match(word1, word2, threshold=0.75):
    """Check if two words are similar enough to count as a match."""
    if word1 == word2:
        return True
    if not word1 or not word2:
        return False
    sim = 1 - Levenshtein.distance(word1, word2) / max(len(word1), len(word2))
    return sim >= threshold

def get_content_words(text):
    """Extract content words (non-function words) from text."""
    words = clean(text).split()
    return [w for w in words if w not in FUNCTION_WORDS and len(w) > 1]


def get_all_words(text):
    """Get all words from cleaned text."""
    return clean(text).split()


def word_overlap(target_words, response_words):
    """Calculate what fraction of target words appear in response (with fuzzy matching)."""
    if not target_words:
        return 0
    matched = sum(1 for w in target_words
                  if any(fuzzy_word_match(w, rw) for rw in response_words))
    return matched / len(target_words)


def content_overlap(target, response):
    """Calculate content word overlap ratio (with fuzzy matching)."""
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
    """Check if response is essentially empty/silence."""
    cleaned = clean(response)
    return len(cleaned) == 0 or cleaned in ['', ' ']


def score_eit(target, response):
    """
    Score an EIT response on 0-4 scale using rule-based approach.

    4 = exact repetition
    3 = meaning preserved, minor grammar changes okay
    2 = more than half of ideas, meaning close but inexact
    1 = about half of ideas, lots missing
    0 = silence, garbled, or minimal repetition
    """
    # Handle empty/missing responses
    if not response or is_empty_response(response):
        return 0

    # Clean both texts
    t_clean = clean(target)
    r_clean = clean(response)

    # Get word lists
    t_words = get_all_words(target)
    r_words = get_all_words(response)
    t_content = get_content_words(target)
    r_content = get_content_words(response)

    # Metrics
    lev_sim = levenshtein_sim(target, response)
    content_ov = content_overlap(target, response)
    word_ov = word_overlap(t_words, r_words)

    r_content_count = sum(1 for w in t_content
                          if any(fuzzy_word_match(w, rw) for rw in r_content))
    total_content = len(t_content)

    # === Score 4: Exact or near-exact match ===
    if lev_sim >= 0.90 and content_ov >= 0.95:
        return 4

    # === Score 3: Meaning preserved, grammar changes okay ===
    # All or nearly all content words present, high similarity
    if content_ov >= 0.80 and lev_sim >= 0.60:
        return 3

    # Content words all there but some reordering/grammar changes
    if content_ov >= 0.85 and word_ov >= 0.60:
        return 3

    # === Score 2: More than half of ideas, meaning close ===
    if content_ov >= 0.50 and lev_sim >= 0.40:
        return 2

    if content_ov >= 0.60 and len(r_words) >= 3:
        return 2

    # === Score 1: About half of ideas, lots missing ===
    if content_ov >= 0.25 and len(r_words) >= 3:
        return 1

    if r_content_count >= 2 and len(r_words) >= 3:
        return 1

    # === Score 0: Minimal or nothing ===
    return 0


# ============================================================
# Test it
# ============================================================

if __name__ == "__main__":
    # Test cases from the rubric examples
    test_cases = [
        # (target, response, expected_approximate_score)
        ("Quiero cortarme el pelo", "Quiero cortarme el pelo", 4),
        ("El carro lo tiene Pedro", "El carro tiene Pedro", 3),
        ("Dudo que sepa manejar muy bien", "Dudo que sepa manajar bien", 3),
        ("El chico con el que yo salgo es español", "El chico con yo salgo ...um.. está bien", 2),
        ("Después de cenar me fui a dormir tranquilo", "Despues de cenar fue en- tranquilo", 1),
        ("Las calles de esta ciudad son muy anchas", "Las calles..es-[gibberish]...", 0),
        ("Quiero cortarme el pelo", "Manaña", 0),
    ]

    print(f"{'Score':>5} | {'Exp':>3} | Target → Response")
    print("-" * 80)
    for target, response, expected in test_cases:
        score = score_eit(target, response)
        match = "✓" if score == expected else "✗"
        print(f"  {score}  {match} | {expected:>3} | {target} → {response}")