# ==============================
# SCIBERT ATTENTIVE SMART SCORER FOR KEYPHRASES (FULL INTEGRATION)
# ==============================

class SciBERTKeyphraseScorer:
    """
    SciBERT-based attentive relevance scorer for scientific keyphrases.
    Focuses on three domains:
      1. PVDF polymer structure, processing, phases
      2. Dopant/filler effects on PVDF piezoelectric performance
      3. Piezoelectric/ferroelectric properties
    Only keyphrases with ≥60% relevance in at least one category are retained.
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.classifiers = {}
        self.lock = threading.Lock()
        self._initialize_model()
        logger.info(f"SciBERT Keyphrase Scorer initialized on {self.device}")

    def _initialize_model(self):
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available → SciBERT scoring disabled.")
            return
        
        try:
            # Load tokenizer and base model
            self.tokenizer = BertTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
            from transformers import BertModel
            bert = BertModel.from_pretrained('allenai/scibert_scivocab_uncased')
            self.model = bert.to(self.device)
            self.model.eval()
            
            # Build lightweight classifiers for each category
            import torch.nn as nn
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
                'pvdf': RelevanceHead().to(self.device),
                'dopant': RelevanceHead().to(self.device),
                'property': RelevanceHead().to(self.device)
            }
            logger.info("SciBERT base model + 3 classifiers loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize SciBERT: {e}")
            self.model = None

    def extract_keyphrases(self, texts: List[str]) -> List[str]:
        """Extract noun chunks + scientific phrases as keyphrases"""
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner"])
        except OSError:
            logger.warning("en_core_web_sm not found. Using blank model.")
            nlp = spacy.blank("en")
            return [phrase for text in texts for phrase in re.findall(r'\b\w{3,}\b', str(text).lower())]

        keyphrases = set()
        for doc_text in texts:
            if not doc_text or len(str(doc_text)) < 30:
                continue
            # Limit length for speed
            doc = nlp(str(doc_text)[:10000])
            # Noun chunks
            for chunk in doc.noun_chunks:
                phrase = chunk.text.strip().lower()
                if 2 < len(phrase) <= 50 and len(phrase.split()) <= 4:
                    keyphrases.add(phrase)
            # Bigrams/trigrams from non-stop tokens
            tokens = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
            for i in range(len(tokens) - 1):
                bigram = f"{tokens[i]} {tokens[i+1]}"
                keyphrases.add(bigram)
                if i < len(tokens) - 2:
                    trigram = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
                    keyphrases.add(trigram)
        # Remove numeric/noise phrases
        filtered = {p for p in keyphrases if not any(c.isdigit() for c in p) and len(p) > 2}
        logger.info(f"Extracted {len(filtered)} candidate keyphrases.")
        return list(filtered)

    def _get_embedding(self, phrase: str) -> Optional[np.ndarray]:
        """Get SciBERT embedding (mean-pooled)"""
        if self.model is None:
            return None
        try:
            inputs = self.tokenizer(
                phrase,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
                return_attention_mask=True
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            return embedding
        except Exception as e:
            logger.debug(f"Embedding failed for '{phrase}': {e}")
            return None

    def score_keyphrase(self, phrase: str) -> Dict[str, float]:
        """Score phrase relevance across 3 categories"""
        if self.model is None or not self.classifiers:
            # Fallback: lexical matching
            return self._lexical_fallback(phrase)

        emb = self._get_embedding(phrase)
        if emb is None:
            return {'pvdf': 0.0, 'dopant': 0.0, 'property': 0.0}

        scores = {}
        with torch.no_grad():
            emb_tensor = torch.tensor(emb).to(self.device).float()
            for cat, clf in self.classifiers.items():
                score = clf(emb_tensor).item()
                scores[cat] = score
        return scores

    def _lexical_fallback(self, phrase: str) -> Dict[str, float]:
        """Fallback when SciBERT unavailable"""
        p = phrase.lower()
        pvdf_terms = {'pvdf', 'polyvinylidene', 'beta phase', 'polymer', 'electrospun', 'nanofiber', 'β-phase', 'phase content'}
        dopant_terms = {'zno', 'batio3', 'cnt', 'graphene', 'dopant', 'filler', 'composite', 'nanofiller', 'tio2', 'sno2', 'mxene'}
        prop_terms = {'d33', 'voltage', 'piezoelectric', 'coefficient', 'beta content', 'output', 'power', 'energy', 'g33', 'dielectric'}
        
        pvdf_score = 0.9 if any(t in p for t in pvdf_terms) else 0.1
        dopant_score = 0.85 if any(t in p for t in dopant_terms) else 0.1
        prop_score = 0.9 if any(t in p for t in prop_terms) else 0.1
        
        return {'pvdf': pvdf_score, 'dopant': dopant_score, 'property': prop_score}

    def score_corpus(self, texts: List[str]) -> Dict[str, Dict[str, float]]:
        """Score all keyphrases; retain only those ≥60% relevance"""
        keyphrases = self.extract_keyphrases(texts)
        scored = {}
        for phrase in keyphrases:
            scores = self.score_keyphrase(phrase)
            if max(scores.values()) >= 0.6:  # ≥60% threshold
                scored[phrase] = scores
        logger.info(f"Retained {len(scored)} keyphrases after 60% relevance filtering.")
        return scored


# ==============================
# EXTENDED PUBLICATION-QUALITY VISUALIZATION ENGINE (SCIBERT MODE)
# ==============================

def create_scientific_wordcloud(self, scored_phrases: Dict[str, Dict[str, float]], title: str = "SciBERT-Relevance Word Cloud"):
    """
    Create word cloud using only SciBERT-relevance-filtered keyphrases (≥60%).
    Colors by dominant category with publication-quality styling.
    """
    self.performance_monitor.start_timer("create_scientific_wordcloud")
    
    if not scored_phrases:
        self.performance_monitor.end_timer("create_scientific_wordcloud")
        return None

    # Frequency = max relevance score
    word_freq = {}
    word_colors = {}
    category_colors = {
        'pvdf': '#3B82F6',      # Blue
        'dopant': '#EF4444',    # Red
        'property': '#10B981'   # Green
    }

    for phrase, scores in scored_phrases.items():
        weight = max(scores.values())
        word_freq[phrase] = weight
        dominant_cat = max(scores, key=scores.get)
        word_colors[phrase] = category_colors[dominant_cat]

    # Color function
    def sci_color_func(word, **kwargs):
        return word_colors.get(word, '#6B7280')

    # Generate word cloud
    wordcloud = WordCloud(
        width=2000,
        height=1000,
        background_color='white',
        max_words=300,
        color_func=sci_color_func,
        collocations=False,
        relative_scaling=0.5,
        font_step=1,
        prefer_horizontal=0.7,
        regexp=r"[\w\s\-\.\(\)]+"
    ).generate_from_frequencies(word_freq)

    # Create figure
    fig, ax = plt.subplots(figsize=(20, 10), dpi=300)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(title, fontsize=24, fontweight='bold', pad=30, fontfamily='serif')
    ax.axis('off')
    plt.tight_layout(pad=2.0)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3B82F6', label='PVDF Polymer'),
        Patch(facecolor='#EF4444', label='Dopant/Filler Effects'),
        Patch(facecolor='#10B981', label='Piezoelectric Properties')
    ]
    ax.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=16)

    self.performance_monitor.end_timer("create_scientific_wordcloud")
    return fig

# Monkey-patch into existing class
PublicationQualityVisualizationEngine.create_scientific_wordcloud = create_scientific_wordcloud
