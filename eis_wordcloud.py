import torch
import torch.nn as nn
import numpy as np
import spacy
import re
import threading
import logging
from transformers import BertTokenizer, BertModel
from typing import List, Dict, Optional
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EISSciBERTKeyphraseScorer:
    """
    SciBERT-based attentive relevance scorer for EIS + AI/ML research.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.classifiers = {}
        self.lock = threading.Lock()
        self._initialize_model()

    def _initialize_model(self):
        try:
            # allenai/scibert_scivocab_uncased is the gold standard for this
            self.tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
            self.model = BertModel.from_pretrained('allenai/scibert_scivocab_uncased').to(self.device)
            self.model.eval()
            
            class RelevanceHead(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.head = nn.Sequential(
                        nn.Linear(768, 256),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256, 1),
                        nn.Sigmoid()
                    )
                def forward(self, x):
                    return self.head(x)
            
            self.classifiers = {
                'eis_physics': RelevanceHead().to(self.device),
                'ai_methods': RelevanceHead().to(self.device),
                'app_results': RelevanceHead().to(self.device)
            }
            logger.info(f"EIS-AI SciBERT Scorer initialized on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize SciBERT: {e}")
            self.model = None

    def extract_keyphrases(self, texts: List[str]) -> List[str]:
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner"])
        except OSError:
            nlp = spacy.blank("en")

        keyphrases = set()
        for doc_text in texts:
            if not doc_text or len(str(doc_text)) < 30: continue
            doc = nlp(str(doc_text)[:10000].lower())
            
            # Noun chunks capture complex terms like 'Equivalent Circuit Model'
            for chunk in doc.noun_chunks:
                if 2 < len(chunk.text) <= 50 and len(chunk.text.split()) <= 4:
                    keyphrases.add(chunk.text.strip())
            
            # Bigrams and Trigrams capture technical pairings
            tokens = [t.text for t in doc if t.is_alpha and not t.is_stop]
            for i in range(len(tokens) - 1):
                keyphrases.add(f"{tokens[i]} {tokens[i+1]}")
                if i < len(tokens) - 2:
                    keyphrases.add(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}")

        return [p for p in keyphrases if not any(c.isdigit() for c in p) and len(p) > 2]

    def _get_embedding(self, phrase: str) -> Optional[torch.Tensor]:
        if self.model is None: return None
        inputs = self.tokenizer(phrase, return_tensors="pt", padding=True, 
                                truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)

    def score_keyphrase(self, phrase: str) -> Dict[str, float]:
        with self.lock: # Ensures thread safety during inference
            if self.model is None: return self._lexical_fallback(phrase)
            emb = self._get_embedding(phrase)
            if emb is None: return {'eis_physics': 0.0, 'ai_methods': 0.0, 'app_results': 0.0}
            
            return {cat: clf(emb).item() for cat, clf in self.classifiers.items()}

    def _lexical_fallback(self, phrase: str) -> Dict[str, float]:
        p = phrase.lower()
        eis_terms = {'eis', 'impedance', 'nyquist', 'bode', 'warburg', 'equivalent circuit'}
        ai_terms = {'machine learning', 'neural network', 'deep learning', 'cnn', 'lstm', 'data-driven'}
        app_terms = {'soh', 'state of health', 'battery', 'rct', 'charge transfer', 'degradation'}
        
        return {
            'eis_physics': 0.9 if any(t in p for t in eis_terms) else 0.1,
            'ai_methods': 0.9 if any(t in p for t in ai_terms) else 0.1,
            'app_results': 0.9 if any(t in p for t in app_terms) else 0.1
        }

    def score_corpus(self, texts: List[str]) -> Dict[str, Dict[str, float]]:
        phrases = self.extract_keyphrases(texts)
        scored = {}
        for phrase in phrases:
            scores = self.score_keyphrase(phrase)
            if max(scores.values()) >= 0.6:
                scored[phrase] = scores
        logger.info(f"Scored {len(scored)} phrases above threshold.")
        return scored

# Visualization Helper
def generate_eis_ai_visual(scored_phrases: Dict[str, Dict[str, float]], title: str = "EIS-AI Knowledge Map"):
    if not scored_phrases: return None

    # Blue = Physics, Purple = AI, Green = Apps
    cat_colors = {'eis_physics': '#1E40AF', 'ai_methods': '#7C3AED', 'app_results': '#059669'}
    
    word_freq = {p: max(s.values()) for p, s in scored_phrases.items()}
    word_colors = {p: cat_colors[max(s, key=s.get)] for p, s in scored_phrases.items()}

    wc = WordCloud(
        width=2000, height=1000, background_color='white',
        color_func=lambda w, **kwargs: word_colors.get(w, '#6B7280'),
        relative_scaling=0.5
    ).generate_from_frequencies(word_freq)

    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title(title, fontsize=24, fontweight='bold', pad=20)
    ax.axis('off')

    legend_elements = [
        Patch(facecolor='#1E40AF', label='EIS Physics'),
        Patch(facecolor='#7C3AED', label='AI/ML Methods'),
        Patch(facecolor='#059669', label='Battery Metrics (SOH/Rct)')
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=3)
    return fig
