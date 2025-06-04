# Supported Spacy pacages
from typing import Union

import qalsadi.lemmatizer
import spacy
import zeyrek
from app.app_data.data.utils import sent_tokenize_sentences, word_tokenize_sentence, detokenize_two_levels
from app.app_data.lemmas.dikta_parser import DictaParser

# ukrainian

_ukrainian_parser = None


def _get_ukrainian_parser():
    # Use
    global _ukrainian_parser
    if _ukrainian_parser is None:  # Initialize only if not already cached
        _ukrainian_parser = spacy.load("uk_core_news_sm")
    return _ukrainian_parser


def lemmatize_ukrainian(text: str) -> str:
    nlp = _get_ukrainian_parser()
    nlp_text = nlp(text)
    return ' '.join([_.lemma_ for _ in nlp_text])


# Chinese

_chinese_parser = None


def _get_chinese_parser():
    # Use
    global _chinese_parser
    if _chinese_parser is None:  # Initialize only if not already cached
        _chinese_parser = spacy.load("uk_core_news_sm")
    return _chinese_parser


def lemmatize_chinese(word: str) -> str:
    nlp = _get_chinese_parser()
    nlp_word = nlp(word)[0]
    return nlp_word.lemma_ if nlp_word.pos_ in ("VERB", "NOUN") else None


# japanese

_japanese_parser = None


def _get_japanese_parser():
    # Use
    global _japanese_parser
    if _japanese_parser is None:  # Initialize only if not already cached
        _japanese_parser = spacy.load("ja_core_news_sm")
    return _japanese_parser


def lemmatize_japanese(word: str) -> str:
    nlp = _get_japanese_parser()
    nlp_word = nlp(word)
    return nlp_word.lemma_


# Spanish


def _get_spanish_parser():
    global _spanish_parser
    if _spanish_parser is None:  # Initialize only if not already cached
        _spanish_parser = spacy.load("es_core_news_md")
    return _spanish_parser


def lemmatize_spanish(text: str) -> str:
    nlp = _get_spanish_parser()
    nlp_text = nlp(text)
    return ' '.join([_.lemma_ for _ in nlp_text])

# Arabic

_arabic_parser = None


def _get_arabic_parser():
    global _arabic_parser
    if _arabic_parser is None:  # Initialize only if not already cached
        _arabic_parser = qalsadi.lemmatizer.Lemmatizer()
    return _arabic_parser


def lemmatize_arabic(word: str) -> str:
    lemmer = _get_arabic_parser()
    return lemmer.lemmatize_text(word)


# Hebrew

_dicta_parser_cache = None
_arabic_lemmer_chache = None
_spanish_parser = None


def _get_dicta_parser():
    global _dicta_parser_cache
    if _dicta_parser_cache is None:  # Initialize only if not already cached
        _dicta_parser_cache = DictaParser()
    return _dicta_parser_cache


def _llematize_word(word: str):
    d = _get_dicta_parser()
    result_dikta = d.parse(word)
    ud_trees = result_dikta["ud_trees"]
    x = ''.join([_['LEMMA'] for _ in ud_trees[0]])
    return x



def lemmatize_hebrew(text: str) -> str:
    parser = _get_dicta_parser()
    parsed_data = parser.parse(text)
    lemma_text = parser.lemmatize(parsed_data)
    return ' '.join(lemma_text)


# Turkish

_turkish_parser = None


def _get_turkish_parser():
    global _turkish_parser
    if _turkish_parser is None:  # Initialize only if not already cached
        _turkish_parser = zeyrek.MorphAnalyzer()
    return _turkish_parser


def lemmatize_turkish(word: str) -> str:
    analyzer = _get_turkish_parser()
    lemmas = analyzer.lemmatize(word)
    return ' '.join(lemma[1][0] for lemma in lemmas)


# indonesian

_indonesian_parser = None


def _get_indonesian_parser():
    global _indonesian_parser
    if _indonesian_parser is None:  # Initialize only if not already cached
        from nlp_id.lemmatizer import Lemmatizer

        _indonesian_parser = Lemmatizer()
    return _indonesian_parser


def lemmatize_indonesian(word: str) -> str:
    lemmatizer = _get_indonesian_parser()
    return lemmatizer.lemmatize(word)


LANGUAGE_TO_lemmatizer = {
    "hebrew": lemmatize_hebrew,
    "turkish": lemmatize_turkish,
    "spanish": lemmatize_spanish,
    "ukrainian": lemmatize_ukrainian,
    "japanese": lemmatize_japanese,
    "arabic": lemmatize_arabic,
    "chinese": lemmatize_chinese,
    "indonesian": lemmatize_indonesian,
}
