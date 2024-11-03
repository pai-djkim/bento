from __future__ import annotations
import bentoml
from transformers import pipeline

@bentoml.service(
    resources={"cpu": "2"},
    traffic={"timeout": 10},
)
class TextClassification:
    def __init__(self) -> None:
        # Load DistilBERT model for text classification
        self.pipeline = pipeline('text-classification', model='distilbert-base-uncased')

    @bentoml.api
    def classify(self, text: str) -> dict:
        result = self.pipeline(text)
        return {"부정": result[0]["score"]}  # Return classification label and score