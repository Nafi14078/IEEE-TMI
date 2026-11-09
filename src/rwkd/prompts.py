"""Class names and text-prompt paraphrases for the zero-shot teacher.

Each *paraphrase set* k is one prompt per class (template k + synonym k).
Agreement across the K sets is the "prompt" part of the reliability score.
Edit these lists at kickoff; re-run `scripts/cache_teacher.py --text-only` afterwards.
"""

CLASS_NAMES = {
    "MEL": ["melanoma", "malignant melanoma", "cutaneous melanoma"],
    "NV": ["melanocytic nevus", "benign mole", "common nevus"],
    "BCC": ["basal cell carcinoma", "basal cell skin cancer", "basalioma"],
    "AK": ["actinic keratosis", "solar keratosis", "precancerous actinic keratosis"],
    "BKL": ["seborrheic keratosis", "benign keratosis", "benign keratosis-like lesion"],
    "SCC": ["squamous cell carcinoma", "cutaneous squamous cell carcinoma",
            "squamous cell skin cancer"],
    "DF": ["dermatofibroma", "benign fibrous histiocytoma", "dermatofibroma skin nodule"],
    "VASC": ["vascular lesion", "angioma", "hemangioma"],
}

TEMPLATES = [
    "a dermoscopic image of {}.",
    "a dermatoscopy photo of {}.",
    "dermoscopy of a skin lesion showing {}.",
    "a close-up image of a skin lesion diagnosed as {}.",
    "an image of {}, a skin lesion.",
    "a clinical photo of {} on the skin.",
    "skin lesion: {}.",
    "this dermoscopic image shows {}.",
]


def build_prompt_sets(classes, templates=TEMPLATES, names=CLASS_NAMES):
    """Return K lists, each with one prompt per class (same class order)."""
    sets = []
    for k, t in enumerate(templates):
        sets.append([t.format(names[c][k % len(names[c])]) for c in classes])
    return sets
