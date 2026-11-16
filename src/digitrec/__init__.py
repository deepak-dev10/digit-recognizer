"""digitrec — handwritten digit recognition.

The package is organised as a five-stage pipeline (see ``plan.md``):

1. Loader        – read an image file, fix EXIF rotation, convert to grayscale.
2. Preprocessor  – make the image look like an MNIST sample (28x28, white on black, centred).
3. Segmenter     – split the image into one crop per digit (single-digit for now).
4. Classifier    – a pluggable ``Predictor`` backend (pretrained in Phase 1, own CNN in Phase 2).
5. Assembler     – join per-digit predictions into the final number with a confidence score.
"""

__version__ = "0.1.0"
