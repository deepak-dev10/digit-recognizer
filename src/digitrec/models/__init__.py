"""Model backends for digit classification.

Every backend implements the :class:`~digitrec.models.base.Predictor` protocol, so the
pipeline can use the Phase 1 pretrained model and the Phase 2 scratch CNN interchangeably.
"""

from digitrec.models.base import NUM_CLASSES, DigitPrediction, Predictor, probs_to_predictions

__all__ = ["NUM_CLASSES", "DigitPrediction", "Predictor", "probs_to_predictions"]
