from typing import List, Dict, Union
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

class SentimentAnalyzer:
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        self.labels = ["negative", "neutral", "positive"]

    def analyze_text(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of a single text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            
        sentiment_scores = {
            label: float(prob)
            for label, prob in zip(self.labels, probabilities[0].cpu().numpy())
        }
        
        return {
            "sentiment": self.labels[probabilities.argmax().item()],
            "scores": sentiment_scores
        }

    def analyze_batch(self, texts: List[str], chunk_size: int = 8) -> List[Dict[str, float]]:
        """Analyze sentiment of multiple texts in batches."""
        results = []
        
        for i in range(0, len(texts), chunk_size):
            batch_texts = texts[i:i + chunk_size]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=1)
            
            batch_results = []
            for probs in probabilities:
                sentiment_scores = {
                    label: float(prob)
                    for label, prob in zip(self.labels, probs.cpu().numpy())
                }
                batch_results.append({
                    "sentiment": self.labels[probs.argmax().item()],
                    "scores": sentiment_scores
                })
            
            results.extend(batch_results)
        
        return results

    def analyze_document(self, chunks: List[str]) -> Dict[str, Union[str, Dict[str, float]]]:
        """Analyze sentiment of a document by analyzing its chunks and aggregating results."""
        chunk_sentiments = self.analyze_batch(chunks)
        
        # Calculate average sentiment scores across all chunks
        avg_scores = {label: 0.0 for label in self.labels}
        for result in chunk_sentiments:
            for label, score in result["scores"].items():
                avg_scores[label] += score
        
        for label in avg_scores:
            avg_scores[label] /= len(chunks)
        
        # Determine overall sentiment
        overall_sentiment = max(avg_scores.items(), key=lambda x: x[1])[0]
        
        return {
            "overall_sentiment": overall_sentiment,
            "average_scores": avg_scores,
            "chunk_sentiments": chunk_sentiments
        } 