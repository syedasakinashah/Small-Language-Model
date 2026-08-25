"""Work out which subject a PDF belongs to, from its own content.

A student uploading "chapter4_final(2).pdf" should not have to tell the app it is
Biology. Scoring subject vocabulary against the document is crude compared to a
classifier, but it needs no model, no download and no memory -- and when it is
unsure it says so rather than filing the PDF somewhere wrong.
"""

from __future__ import annotations

import re
from collections import Counter

# Terms chosen to be load-bearing for one subject and rare in the others.
SUBJECT_TERMS: dict[str, set[str]] = {
    "Physics": {
        "velocity", "acceleration", "newton", "momentum", "kinetic", "friction",
        "joule", "voltage", "ampere", "circuit", "refraction", "wavelength",
        "thermodynamics", "inertia", "displacement", "electromagnetic", "torque",
        "capacitor", "resistor", "amplitude", "gravitational",
    },
    "Chemistry": {
        "molecule", "compound", "valency", "isotope", "covalent", "ionic",
        "catalyst", "oxidation", "reduction", "periodic", "alkali", "acidic",
        "hydrocarbon", "titration", "electrolysis", "stoichiometry", "reagent",
        "molar", "solvent", "solute", "hydroxide",
    },
    "Biology": {
        "cell", "organism", "photosynthesis", "enzyme", "chromosome", "protein",
        "mitochondria", "respiration", "tissue", "nucleus", "genetic", "species",
        "bacteria", "membrane", "digestion", "hormone", "ecosystem", "chlorophyll",
        "reproduction", "evolution", "neuron",
    },
    "Mathematics": {
        "theorem", "equation", "integral", "derivative", "polynomial", "matrix",
        "algebra", "geometry", "trigonometry", "logarithm", "quadratic", "vector",
        "coefficient", "factorise", "hypotenuse", "calculus", "denominator",
        "numerator", "cosine", "sine",
    },
    "Computer Science": {
        "algorithm", "compiler", "database", "variable", "recursion", "binary",
        "array", "boolean", "syntax", "programming", "software", "hardware",
        "processor", "memory", "network", "encryption", "pseudocode", "iteration",
        "debugging", "operating",
    },
    "Economics": {
        "demand", "supply", "inflation", "market", "elasticity", "revenue",
        "monopoly", "gdp", "fiscal", "monetary", "consumer", "producer",
        "equilibrium", "opportunity", "scarcity", "taxation", "subsidy",
        "unemployment", "trade",
    },
    "History": {
        "empire", "dynasty", "revolution", "treaty", "colonial", "century",
        "independence", "partition", "civilisation", "monarchy", "conquest",
        "reign", "settlement", "war", "movement", "reform",
    },
    "Geography": {
        "climate", "erosion", "latitude", "longitude", "monsoon", "plateau",
        "tectonic", "rainfall", "population", "urbanisation", "glacier",
        "vegetation", "irrigation", "terrain", "delta",
    },
    "English": {
        "grammar", "noun", "verb", "adjective", "metaphor", "simile", "clause",
        "narrator", "poem", "stanza", "punctuation", "vocabulary", "comprehension",
        "essay", "paragraph", "tense",
    },
    "Islamiat": {
        "quran", "hadith", "prophet", "surah", "islam", "muslim", "prayer",
        "zakat", "ramadan", "sunnah", "caliph", "faith", "pillars",
    },
    "Statistics": {
        "mean", "median", "variance", "deviation", "probability", "distribution",
        "sample", "regression", "correlation", "hypothesis", "frequency",
        "quartile", "histogram",
    },
}

# A document must show a real spread of a subject's vocabulary, not one stray
# word: "vocabulary" alone should never file a document under English.
_MIN_DISTINCT_TERMS = 4
_MIN_SCORE = 15
_MIN_MARGIN = 1.4


def _word_counts(text: str) -> Counter[str]:
    """Count words, folding simple plurals so 'enzymes' matches the term 'enzyme'."""
    counts: Counter[str] = Counter()
    for word in re.findall(r"[a-z]{3,}", text):
        counts[word] += 1
        if word.endswith("ies") and len(word) > 4:
            counts[word[:-3] + "y"] += 1
        elif word.endswith("es") and len(word) > 4:
            counts[word[:-2]] += 1
            counts[word[:-1]] += 1
        elif word.endswith("s") and len(word) > 3:
            counts[word[:-1]] += 1
    return counts


def detect_subject(doc, max_pages: int = 12) -> tuple[str | None, float]:
    """Guess the subject of a document.

    Returns ``(subject, confidence)``; ``subject`` is ``None`` when the content
    doesn't clearly belong to any known subject -- better to ask the student than
    to file their chemistry notes under history.
    """
    sample = " ".join(p.text for p in doc.pages[:max_pages]).lower()
    words = _word_counts(sample)
    if not words:
        return None, 0.0

    scores: dict[str, int] = {}
    distincts: dict[str, int] = {}
    for subject, terms in SUBJECT_TERMS.items():
        # weight distinct terms heavily: one word repeated across a long PDF
        # ("cell" in a spreadsheet manual) must not carry a whole subject
        hits = sum(words[t] for t in terms)
        distinct = sum(1 for t in terms if words[t])
        distincts[subject] = distinct
        scores[subject] = hits + 3 * distinct

    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best, best_score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0

    if distincts[best] < _MIN_DISTINCT_TERMS or best_score < _MIN_SCORE:
        return None, 0.0
    if best_score < runner_up * _MIN_MARGIN:
        return None, 0.0

    confidence = min(1.0, best_score / (best_score + runner_up + 8))
    return best, round(confidence, 2)


def detect_from_filename(name: str) -> str | None:
    """Fall back to the filename: 'physics_ch4.pdf' is a strong hint."""
    stem = re.sub(r"[^a-z]+", " ", name.lower())
    for subject in SUBJECT_TERMS:
        if subject.lower() in stem:
            return subject
    aliases = {"maths": "Mathematics", "math": "Mathematics", "bio": "Biology",
               "chem": "Chemistry", "phy": "Physics", "cs": "Computer Science",
               "econ": "Economics", "geo": "Geography", "stats": "Statistics"}
    for token in stem.split():
        if token in aliases:
            return aliases[token]
    return None
