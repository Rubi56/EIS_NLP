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

# Assume TRANSFORMERS_AVAILABLE and logger are defined globally in your environment
logger = logging.getLogger(__name__)

class EISSciBERTKeyphraseScorer:
    """
    SciBERT-based attentive relevance scorer adapted for EIS + AI/ML research.
    Focuses on:
      1. EIS Core: Physics, Nyquist/Bode plots, Equivalent Circuits
      2. AI/ML: Data-driven methods, Neural Networks, Feature Extraction
      3. Apps/Params: Battery SOH, SOC, Rct, degradation parameters
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.classifiers = {}
        self.lock = threading.Lock()
        self._initialize_model()
        logger.info(f"EIS-AI SciBERT Scorer initialized on {self.device}")

    def _initialize_model(self):
        try:
            # Load SciBERT components
            self.tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
            bert = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
            self.model = bert.to(self.device)
            self.model.eval()
            
            # Neural head architecture for classification
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
            
            # Updated categories for EIS and AI
            self.classifiers = {
                'eis_physics': RelevanceHead().to(self.device),
                'ai_methods': RelevanceHead().to(self.device),
                'app_results': RelevanceHead().to(self.device)
            }
            logger.info("EIS-AI SciBERT base model + 3 specialized heads loaded.")
        except Exception as e:
            logger.error(f"Failed to initialize SciBERT for EIS: {e}")
            self.model = None

    def extract_keyphrases(self, texts: List[str]) -> List[str]:
        """Extract noun chunks + scientific bigrams/trigrams"""
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner"])
        except OSError:
            nlp = spacy.blank("en")

        keyphrases = set()
        for doc_text in texts:
            if not doc_text or len(str(doc_text)) < 30:
                continue
            doc = nlp(str(doc_text)[:10000].lower())
            
            # Extract Noun chunks
            for chunk in doc.noun_chunks:
                phrase = chunk.text.strip()
                if 2 < len(phrase) <= 50 and len(phrase.split()) <= 4:
                    keyphrases.add(phrase)
            
            # Extract n-grams from meaningful tokens
            tokens = [t.text for t in doc if t.is_alpha and not t.is_stop]
            for i in range(len(tokens) - 1):
                keyphrases.add(f"{tokens[i]} {tokens[i+1]}") # Bigram
                if i < len(tokens) - 2:
                    keyphrases.add(f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}") # Trigram

        # Filter: No digits, must be > 2 chars
        filtered = {p for p in keyphrases if not any(c.isdigit() for c in p) and len(p) > 2}
        return list(filtered)

    def _get_embedding(self, phrase: str) -> Optional[torch.Tensor]:
        if self.model is None: return None
        inputs = self.tokenizer(phrase, return_tensors="pt", padding=True, 
                                truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)

    def score_keyphrase(self, phrase: str) -> Dict[str, float]:
        if self.model is None:
            return self._lexical_fallback(phrase)

        emb = self._get_embedding(phrase)
        if emb is None:
            return {'eis_physics': 0.0, 'ai_methods': 0.0, 'app_results': 0.0}

        scores = {}
        with torch.no_grad():
            for cat, clf in self.classifiers.items():
                scores[cat] = clf(emb).item()
        return scores

    def _lexical_fallback(self, phrase: str) -> Dict[str, float]:
        """Fallback specifically for EIS and AI core terms."""
        p = phrase.lower()
        # EIS Physics
        eis_terms = {'eis', 'impedance', 'nyquist', 'bode', 'warburg', 'equivalent circuit', 'spectroscopy'}
        # AI/ML Methods
        ai_terms = {'machine learning', 'neural network', 'deep learning', 'cnn', 'lstm', 'data-driven', 'predictive'}
        # Applications/Results
        app_terms = {'soh', 'state of health', 'battery', 'rct', 'charge transfer', 'degradation', 'resistance'}
        
        return {
            'eis_physics': 0.9 if any(t in p for t in eis_terms) else 0.1,
            'ai_methods': 0.9 if any(t in p for t in ai_terms) else 0.1,
            'app_results': 0.9 if any(t in p for t in app_terms) else 0.1
        }

    def score_corpus(self, texts: List[str]) -> Dict[str, Dict[str, float]]:
        keyphrases = self.extract_keyphrases(texts)
        scored = {}
        for phrase in keyphrases:
            scores = self.score_keyphrase(phrase)
            if max(scores.values()) >= 0.6:
                scored[phrase] = scores
        return scored

def create_eis_scientific_wordcloud(self, scored_phrases: Dict[str, Dict[str, float]], title: str = "EIS & AI/ML Knowledge Convergence"):
    """
    Generates word cloud for EIS-AI papers. 
    Colors: Blue (Physics), Purple (AI), Green (Apps).
    """
    if hasattr(self, 'performance_monitor'):
        self.performance_monitor.start_timer("create_eis_wordcloud")
    
    if not scored_phrases:
        return None

    word_freq = {}
    word_colors = {}
    # Professional Palette
    category_colors = {
        'eis_physics': '#1E40AF',   # Blue
        'ai_methods': '#7C3AED',    # Purple
        'app_results': '#059669'    # Emerald Green
    }

    for phrase, scores in scored_phrases.items():
        weight = max(scores.values())
        word_freq[phrase] = weight
        dominant_cat = max(scores, key=scores.get)
        word_colors[phrase] = category_colors[dominant_cat]

    def sci_color_func(word, **kwargs):
        return word_colors.get(word, '#6B7280')

    wordcloud = WordCloud(
        width=2000, height=1000,
        background_color='white',
        max_words=300,
        color_func=sci_color_func,
        collocations=False,
        relative_scaling=0.5,
        font_step=1
    ).generate_from_frequencies(word_freq)

    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=28, fontweight='bold', pad=30, fontfamily='serif')
    ax.axis('off')

    # Legend for scientific context
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1E40AF', label='EIS Physics & Spectroscopy'),
        Patch(facecolor='#7C3AED', label='AI/ML Methodologies'),
        Patch(facecolor='#059669', label='Battery Apps & Parameters (SOH/Rct)')
    ]
    ax.legend(handles=legend_elements, loc='lower center', 
              bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=16, frameon=False)

    plt.tight_layout()
    
    if hasattr(self, 'performance_monitor'):
        self.performance_monitor.end_timer("create_eis_wordcloud")
    return fig

# Patching into your existing visualization class
PublicationQualityVisualizationEngine.create_scientific_wordcloud = create_eis_scientific_wordcloud
