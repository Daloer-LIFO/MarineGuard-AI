
from backend.services.classifier import classify_incident


def test_classifier_returns_expected_fields():
    result = classify_incident(
        "A cargo ship collided with another vessel."
    )

    assert "category" in result
    assert "confidence" in result
    assert 0 <= result["confidence"] <= 1


def test_classifier_rejects_short_input():
    try:
        classify_incident("Help")
    except ValueError:
        return

    raise AssertionError("Short descriptions should be rejected.")
