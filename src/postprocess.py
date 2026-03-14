"""Match transcribed segments to EIT target sentences."""

import Levenshtein

TARGETS = [
    "Quiero cortarme el pelo",
    "El libro está en la mesa",
    "El carro lo tiene Pedro",
    "El se ducha cada mañana",
    "Qué dice usted que va a hacer hoy",
    "Dudo que sepa manejar muy bien",
    "Las calles de esta ciudad son muy anchas",
    "Puede que llueva mañana todo el día",
    "Las casas son muy bonitas pero caras",
    "Me gustan las películas que acaban bien",
    "El chico con el que yo salgo es español",
    "Después de cenar me fui a dormir tranquilo",
    "Quiero una casa en la que vivan mis animales",
    "A nosotros nos fascinan las fiestas grandiosas",
    "Ella sólo bebe cerveza y no come nada",
    "Me gustaría que el precio de las casas bajara",
    "Cruza a la derecha y después sigue todo recto",
    "Ella ha terminado de pintar su apartamento",
    "Me gustaría que empezara a hacer más calor pronto",
    "El niño al que se le murió el gato está triste",
    "Una amiga mía cuida a los niños de mi vecino",
    "El gato que era negro fue perseguido por el perro",
    "Antes de poder salir él tiene que limpiar su cuarto",
    "La cantidad de personas que fuman ha disminuido",
    "Después de llegar a casa del trabajo tomé la cena",
    "El ladrón al que atrapó la policía era famoso",
    "Le pedí a un amigo que me ayudara con la tarea",
    "El examen no fue tan difícil como me habían dicho",
    "Serías tan amable de darme el libro que está en la mesa",
    "Hay mucha gente que no toma nada para el desayuno",
]


def clean_text(text):
    """Normalize text for comparison."""
    return (text.lower()
            .replace("¿", "").replace("?", "")
            .replace(".", "").replace(",", "")
            .replace("…", "").replace("...", "")
            .strip())


def similarity(a, b):
    """Normalized Levenshtein similarity (1.0 = identical)."""
    a, b = clean_text(a), clean_text(b)
    if not a or not b:
        return 0
    return 1 - Levenshtein.distance(a, b) / max(len(a), len(b))


def match_segments_to_targets(segments, min_similarity=0.25):
    """
    Match transcribed segments to the 30 EIT target sentences.

    Uses greedy best-match assignment: scores every segment against
    every target, then assigns the highest-similarity pairs first,
    ensuring each target and segment is used only once.

    Args:
        segments: List of segment dicts from transcription
        min_similarity: Minimum similarity to consider a match

    Returns:
        List of 30 result dicts with item, target, transcription, similarity
    """
    # Score every segment against every target
    scores = []
    for seg in segments:
        for t_idx, target in enumerate(TARGETS):
            sim = similarity(seg["text"], target)
            scores.append((sim, t_idx, seg))

    scores.sort(key=lambda x: x[0], reverse=True)

    # Greedy assignment
    assigned = {}
    used_segments = set()

    for sim, t_idx, seg in scores:
        seg_key = (seg["start"], seg["end"])
        if sim < min_similarity:
            continue
        if seg_key in used_segments:
            continue
        if t_idx not in assigned:
            assigned[t_idx] = (sim, seg)
            used_segments.add(seg_key)

    # Build results
    results = []
    for t_idx in range(30):
        if t_idx in assigned:
            sim, seg = assigned[t_idx]
            results.append({
                "item": t_idx + 1,
                "target": TARGETS[t_idx],
                "transcription": seg["text"],
                "similarity": round(sim, 3),
                "start": seg["start"],
                "end": seg["end"],
            })
        else:
            results.append({
                "item": t_idx + 1,
                "target": TARGETS[t_idx],
                "transcription": "[no response detected]",
                "similarity": 0,
                "start": None,
                "end": None,
            })

    return results