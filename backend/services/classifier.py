
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Small illustrative dataset for the initial prototype.
# Replace or expand this with a properly labeled dataset
# before reporting final evaluation results.

TRAINING_DATA = [
    ("A cargo ship struck another vessel", "collision"),
    ("Two boats collided near the harbor", "collision"),
    ("A vessel hit a fishing boat", "collision"),
    ("A ship grounded on a submerged reef", "grounding"),
    ("The vessel ran aground in shallow water", "grounding"),
    ("A boat became stranded on a sandbank", "grounding"),
    ("A passenger vessel requested emergency rescue", "distress"),
    ("A ship sent a distress call after engine failure", "distress"),
    ("Crew members requested urgent assistance at sea", "distress"),
    ("An oil spill was reported near the coast", "environmental"),
    ("Fuel leaked into the surrounding seawater", "environmental"),
    ("Floating waste polluted the coastal water", "environmental"),
    ("A vessel completed its scheduled harbor visit", "other"),
    ("Routine maintenance was carried out on a ship", "other"),
    ("A cargo vessel departed from the port", "other"),
]

classifier = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2)
    )),
    ("model", LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ))
])

classifier.fit(
    [item[0] for item in TRAINING_DATA],
    [item[1] for item in TRAINING_DATA]
)


def classify_incident(description: str) -> dict:
    """Classify a maritime incident description."""

    description = description.strip()

    if len(description) < 10:
        raise ValueError(
            "Please provide a more detailed incident description."
        )

    probabilities = classifier.predict_proba([description])[0]
    best_index = probabilities.argmax()

    return {
        "category": classifier.classes_[best_index],
        "confidence": round(float(probabilities[best_index]), 4),
        "note": (
            "Prototype prediction. Validate against real, "
            "representative incident data before operational use."
        )
    }
