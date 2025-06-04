import spacy
from tokenizers.decoders import WordPiece
from transformers import pipeline

# ukrainian
_ukrainian_model_cache = None
_chinese_model_cache = None
_japanese_model_cache = None
_spanish_model_cache = None
_arabic_model_cache = None
_heb_model_cache = None
_turkish_model_cache = None
_indonesian_model_cache = None
_yoruba_model_cache = None


def _get_ukrainian_model():
    global _ukrainian_model_cache
    if _ukrainian_model_cache is None:  # Load model only if it's not already cached
        _ukrainian_model_cache = spacy.load("uk_core_news_sm")
    return _ukrainian_model_cache


def ner_ukrainian(sentence: str) -> str:
    nlp = _get_ukrainian_model()
    print(nlp(sentence).ents)
    ents = nlp(sentence).ents
    return [(ent.label_, ent.text) for ent in ents]


# Chinese
def _get_chinese_model():
    global _chinese_model_cache
    if _chinese_model_cache is None:  # Load model only if it's not already cached
        _chinese_model_cache = spacy.load("zh_core_web_sm")
    return _chinese_model_cache


def ner_chinese(sentence: str) -> str:
    nlp = _get_chinese_model()
    ents = nlp(sentence).ents
    return [(ent.label_, ent.text) for ent in ents]


def _get_japanese_model():
    global _japanese_model_cache
    if _japanese_model_cache is None:  # Load model only if it's not already cached
        _japanese_model_cache = spacy.load("ja_core_news_sm")
    return _japanese_model_cache


# japanese


def ner_japanese(sentence: str) -> str:
    nlp = _get_japanese_model()
    ents = nlp(sentence).ents
    return [(ent.label_, ent.text) for ent in ents]


# Spanish


def _get_spanish_model():
    global _spanish_model_cache
    if _spanish_model_cache is None:
        _spanish_model_cache = spacy.load("es_core_news_sm")
    return _spanish_model_cache


def ner_spanish(sentence: str) -> str:
    nlp = _get_spanish_model()
    ents = nlp(sentence).ents
    return [(ent.label_, ent.text) for ent in ents]


# Arabic


def _get_arabic_model():
    global _arabic_model_cache
    if _arabic_model_cache is None:
        from transformers import pipeline

        _arabic_model_cache = pipeline(
            "ner",
            model="CAMeL-Lab/bert-base-arabic-camelbert-msa-ner",
            aggregation_strategy="simple",
        )
        from tokenizers.decoders import WordPiece

        _arabic_model_cache.tokenizer.backend_tokenizer.decoder = WordPiece()
    return _arabic_model_cache


def ner_arabic(text: str) -> str:
    model = _get_arabic_model()
    output_model = model(text)
    return [(ent["entity_group"], ent["word"]) for ent in output_model]


# Hebrew


def _get_hebrew_model():
    global _heb_model_cache
    if _heb_model_cache is None:  # Load model only if it's not already cached
        oracle = pipeline(
            "ner", model="dicta-il/dictabert-large-ner", aggregation_strategy="simple"
        )
        oracle.tokenizer.backend_tokenizer.decoder = WordPiece()
        _heb_model_cache = oracle
    return _heb_model_cache


def ner_hebrew(text: str) -> str:
    oracle = _get_hebrew_model()
    output_model = oracle(text)
    return [(ent["entity_group"], ent["word"]) for ent in output_model]


# Yoruba


# Turkish


def _get_turkish_model():
    global _turkish_model_cache
    if _turkish_model_cache is None:  # Load model only if it's not already cached
        oracle = pipeline(
            "ner",
            model="akdeniz27/bert-base-turkish-cased-ner",
            aggregation_strategy="simple",
        )
        oracle.tokenizer.backend_tokenizer.decoder = WordPiece()
        _turkish_model_cache = oracle
    return _turkish_model_cache


def ner_turkish(sentence: str) -> dict:
    ner = _get_turkish_model()
    output_model = ner(sentence)
    return [(ent["entity_group"], ent["word"]) for ent in output_model]


# indonesian


def _get_indonesian_model():
    global _indonesian_model_cache
    if _indonesian_model_cache is None:  # Load model only if it's not already cached
        oracle = pipeline(
            "ner", model="cahya/bert-base-indonesian-NER", aggregation_strategy="simple"
        )
        oracle.tokenizer.backend_tokenizer.decoder = WordPiece()
        _indonesian_model_cache = oracle
    return _indonesian_model_cache


def ner_indonesian(text: str) -> str:
    ner = _get_indonesian_model()
    output_model = ner(text)
    return [(ent["entity_group"], ent["word"]) for ent in output_model]


def _get_yoruba_model():
    global _yoruba_model_cache
    if _yoruba_model_cache is None:  # Load model only if it's not already cached
        oracle = pipeline(
            "ner",
            model="mbeukman/xlm-roberta-base-finetuned-yoruba-finetuned-ner-yoruba",
            aggregation_strategy="simple",
        )
        oracle.tokenizer.backend_tokenizer.decoder = WordPiece()
        _yoruba_model_cache = oracle
    return _yoruba_model_cache


def ner_yoruba(text: str) -> str:
    ner = _get_yoruba_model()
    output_model = ner(text)
    return [(ent["entity_group"], ent["word"]) for ent in output_model]


LANGUAGE_TO_NER = {
    "hebrew": ner_hebrew,
    "turkish": ner_turkish,
    "spanish": ner_spanish,
    "ukrainian": ner_ukrainian,
    "japanese": ner_japanese,
    "arabic": ner_arabic,
    "chinese": ner_chinese,
    "indonesian": ner_indonesian,
    "yoruba": ner_yoruba,
}
