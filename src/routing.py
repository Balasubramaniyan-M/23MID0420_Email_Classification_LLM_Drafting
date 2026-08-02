"""
Prediction signal calculation, decision margin estimation, and human review routing logic.
"""

from typing import Dict, Any
import numpy as np
from sklearn.pipeline import Pipeline


def get_prediction_signal(model: Pipeline, text: str) -> Dict[str, Any]:
    """
    Calculate predicted class, confidence/margin signal, and signal type for an input text.

    Parameters
    ----------
    model : Pipeline
        Fitted scikit-learn classification pipeline.
    text : str
        Input string ('subject: ... \nbody: ...').

    Returns
    -------
    Dict[str, Any]
        Dictionary containing predicted_class, signal score, margin score, and signal_type.
    """
    predicted = model.predict([text])[0]
    classifier = model.named_steps["classifier"]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
        order = np.argsort(probabilities)[::-1]
        signal = float(probabilities[order[0]])
        margin = float(probabilities[order[0]] - probabilities[order[1]]) if len(order) > 1 else 1.0
        signal_type = "probability"
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function([text]))
        if scores.ndim == 1:
            margin = float(abs(scores[0]))
            signal = margin
        else:
            top_two = np.sort(scores[0])[-2:]
            margin = float(top_two[1] - top_two[0])
            signal = float(top_two[1])
        signal_type = "decision_score"
    else:
        signal = np.nan
        margin = np.nan
        signal_type = "unavailable"

    return {
        "predicted_class": str(predicted),
        "signal": signal,
        "margin": margin,
        "signal_type": signal_type,
    }


def classify_email(model: Pipeline, subject: str, body: str) -> Dict[str, Any]:
    """
    Classify an email given raw subject and body text.

    Parameters
    ----------
    model : Pipeline
        Fitted intent classification pipeline.
    subject : str
        Email subject text.
    body : str
        Email body text.

    Returns
    -------
    Dict[str, Any]
        Dictionary combining classification signal outputs and original input text.
    """
    text = f"subject: {subject.strip()}\nbody: {body.strip()}"
    result = get_prediction_signal(model, text)
    result.update({
        "subject": subject,
        "body": body,
        "text": text,
    })
    return result


def evaluate_routing_policy(prediction: Dict[str, Any], margin_threshold: float = 0.15) -> bool:
    """
    Apply review routing rule. Flags mandatory_review = True when decision margin is below threshold
    OR predicted class is 'urgent_action'.

    Parameters
    ----------
    prediction : Dict[str, Any]
        Prediction record containing 'margin' and 'predicted_class'.
    margin_threshold : float, optional
        Margin uncertainty cutoff, defaults to 0.15.

    Returns
    -------
    bool
        True if mandatory human review is required, False otherwise.
    """
    margin = prediction.get("margin", 1.0)
    predicted_class = prediction.get("predicted_class", "")

    low_margin = np.isfinite(margin) and margin < margin_threshold
    mandatory_review = low_margin or (predicted_class == "urgent_action")
    return bool(mandatory_review)
