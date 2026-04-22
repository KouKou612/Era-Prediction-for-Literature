import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


POS_TAGS = [
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
]


def spacy_model_available(model_name="en_core_web_sm"):
    try:
        import spacy
        spacy.load(model_name, disable=["ner", "lemmatizer", "attribute_ruler"])
        return True
    except Exception:
        return False


class SyntacticFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="en_core_web_sm", batch_size=32, max_chars=120000):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_chars = max_chars
        self.nlp = None

    def _load_model(self):
        if self.nlp is not None:
            return

        import spacy

        self.nlp = spacy.load(
            self.model_name,
            disable=["ner", "lemmatizer", "attribute_ruler"],
        )

        # if parser is not there, add a simple sentencizer
        if "parser" not in self.nlp.pipe_names and "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer")

    def fit(self, X, y=None):
        self._load_model()
        return self

    def get_feature_names_out(self, input_features=None):
        names = [f"pos_frac_{tag}" for tag in POS_TAGS]
        names += ["avg_token_len", "avg_sent_len"]
        return np.array(names, dtype=object)

    def _extract_one_doc(self, doc):
        pos_counts = {tag: 0 for tag in POS_TAGS}
        token_lens = []
        sent_lens = []

        real_tokens = [tok for tok in doc if not tok.is_space]

        if len(real_tokens) == 0:
            return np.zeros(len(POS_TAGS) + 2, dtype=float)

        for tok in real_tokens:
            if tok.pos_ in pos_counts:
                pos_counts[tok.pos_] += 1
            token_lens.append(len(tok.text))

        total_tokens = len(real_tokens)

        pos_features = [pos_counts[tag] / total_tokens for tag in POS_TAGS]

        for sent in doc.sents:
            sent_tokens = [tok for tok in sent if not tok.is_space]
            if len(sent_tokens) > 0:
                sent_lens.append(len(sent_tokens))

        avg_token_len = float(np.mean(token_lens)) if token_lens else 0.0
        avg_sent_len = float(np.mean(sent_lens)) if sent_lens else 0.0

        return np.array(pos_features + [avg_token_len, avg_sent_len], dtype=float)

    def transform(self, X):
        self._load_model()

        texts = []
        for text in X:
            text = str(text)
            if len(text) > self.max_chars:
                text = text[: self.max_chars]
            texts.append(text)

        features = []
        for doc in self.nlp.pipe(texts, batch_size=self.batch_size):
            features.append(self._extract_one_doc(doc))

        if not features:
            return np.zeros((0, len(POS_TAGS) + 2), dtype=float)
        return np.vstack(features)