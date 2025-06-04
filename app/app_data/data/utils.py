import copy
import itertools
import random
from enum import Enum

from pydantic import BaseModel, Field
from typing import Union, List, Tuple
import pandas as pd
from app.app_data.lemmas.lemmatizers import LANGUAGE_TO_lemmatizer
from app.app_data.lemmas.ner import LANGUAGE_TO_NER
from app.app_data.lemmas.conjuctors import replace_conjunctions


def sent_tokenize_sentences(text: str, language: str) -> List[str]:
    if (
        language == "chinese"
        or language == "japanese"
        or language == "chinese_simplified"
    ):
        from HanziNLP import sentence_segment

        tokenized_sentences = sentence_segment(text)
    else:
        from nltk.tokenize import sent_tokenize

        tokenized_sentences = sent_tokenize(text)
    return tokenized_sentences


def word_tokenize_sentence(sentence: str, language: str) -> str:
    if (
        language == "chinese"
        or language == "japanese"
        or language == "chinese_simplified"
    ):
        from HanziNLP import word_tokenize

        tokenized_sentence = word_tokenize(sentence)
    else:
        from nltk.tokenize import word_tokenize

        tokenized_sentence = word_tokenize(sentence)
    return tokenized_sentence


def detokenize_one_level(tokenized_sentences: str) -> Union[str, List[str]]:
    from nltk.tokenize.treebank import TreebankWordDetokenizer as Detok

    detokenizer = Detok()
    text = detokenizer.detokenize(tokenized_sentences)
    return text


def detokenize_two_levels(tokenized_sentences: List[List[str]]) -> str:
    return detokenize_one_level(
        [detokenize_one_level(sentence) for sentence in tokenized_sentences]
    )


def replace_random_words_with_lemmas(
    tokenized_words: Union[List[str], List[List[str]]],
    num_to_replace: int,
    language: str,
) -> Union[List[str], List[List[str]]]:
    lemmatize = LANGUAGE_TO_lemmatizer[language]
    words = copy.deepcopy(tokenized_words)
    punctuation_marks = {".", "?", "!", "``", "''", ","}
    all_indices = [
        (i, j)
        for i, sublist in enumerate(words)
        for j, word in enumerate(sublist)
        if word not in punctuation_marks and not isinstance(word, int)
    ]
    indices_to_replace = random.sample(
        all_indices, min(num_to_replace, len(all_indices))
    )
    for i, j in indices_to_replace:
        lemmatized_word = lemmatize(words[i][j])
        words[i][j] = lemmatized_word if lemmatized_word is not None else words[i][j]
    return words


def remove_overlapping_words(
    sentences: list, human_summary: str, num_words: int, language: str
):
    punctuation_marks = {".", "?", "!", "``", "''", ","}

    # Normalize the human summary to lowercase and split into words
    summary_words = list(
        set(
            list(
                itertools.chain(
                    [
                        word_tokenize_sentence(sent, language=language)
                        for sent in sent_tokenize_sentences(human_summary, language)
                    ]
                )
            )[0]
        )
    )

    # Flatten the sentences into a list of (word, sentence_index, word_index)
    all_words = [
        (word, sentence_idx, word_idx)
        for sentence_idx, sentence in enumerate(sentences)
        for word_idx, word in enumerate(word_tokenize_sentence(sentence, language))
        if word.lower() in summary_words and word not in punctuation_marks
    ]

    # Randomly sample words to remove from the overlapping words
    words_to_remove = random.sample(all_words, min(num_words, len(all_words)))

    # Convert sampled words into a set of (sentence_index, word_index) for quick lookup
    indices_to_remove = {
        (sentence_idx, word_idx) for _, sentence_idx, word_idx in words_to_remove
    }

    # Reconstruct the sentences without the removed words
    modified_sentences = [
        [
            word
            for word_idx, word in enumerate(word_tokenize_sentence(sentence, language))
            if (sentence_idx, word_idx) not in indices_to_remove
        ]
        for sentence_idx, sentence in enumerate(sentences)
    ]

    return modified_sentences


def shuffle_pairs(tokenized_sentences):
    tokenized_sentences_copy = copy.deepcopy(tokenized_sentences)
    try:
        i = random.randint(0, len(tokenized_sentences) - 2)
        tokenized_sentences_copy[i], tokenized_sentences_copy[i + 1] = (
            tokenized_sentences[i + 1],
            tokenized_sentences[i],
        )
        return tokenized_sentences_copy
    except Exception:
        return tokenized_sentences_copy


def insert_random_sentence(
    tokenized_sentences: List[str], language: str, num_sentences: int = 1
):
    tokenized_sentences_copy = copy.deepcopy(tokenized_sentences)

    # Read the corresponding CSV file
    data = pd.read_csv(
        f"/Users/itaimondshine/PycharmProjects/NLP/eval_metrics/app/app_data/xlsum/{language}.csv"
    )

    # Sample a random text from the dataset
    tokenized_text = sent_tokenize_sentences(data.sample().iloc[0]["text"], language)

    # Ensure we do not try to insert more sentences than are available
    num_sentences = min(num_sentences, len(tokenized_text))

    for _ in range(num_sentences):
        selected_sentence = tokenized_text[random.randint(0, len(tokenized_text) - 1)]
        random_index = random.randint(
            0, len(tokenized_sentences_copy)
        )  # Insert at a random index
        tokenized_sentences_copy.insert(random_index, selected_sentence)

    return tokenized_sentences_copy


import random
from typing import List, Tuple

def swap_entities_in_text(
    text: str,
    ner_results: List[Tuple[str, str]],
    ner_results_full: List[Tuple[str, str]],
    n: int
) -> str:
    """
    Randomly swaps N pairs of entities in the text with entities from the full NER results.

    Args:
        text (str): The input text.
        ner_results (List[Tuple[str, str]]): List of tuples containing labels and entities from the input text.
        ner_results_full (List[Tuple[str, str]]): List of tuples containing all available labels and entities.
        n (int): Number of pairs to swap.

    Returns:
        str: The text with swapped entities.
    """
    # Group entities from the full NER results by their label
    label_to_entities_full = {}
    for label, entity in ner_results_full:
        if label not in label_to_entities_full:
            label_to_entities_full[label] = []
        label_to_entities_full[label].append(entity)

    # Ensure no duplicates in ner_results
    ner_results = list(set(ner_results))

    # Group entities from the ner_results by their label
    label_to_entities = {}
    for label, entity in ner_results:
        if label not in label_to_entities:
            label_to_entities[label] = []
        label_to_entities[label].append(entity)

    # Identify labels with at least one entity in ner_results and at least two entities in the full results
    swappable_labels = [
        label for label, entities in label_to_entities.items()
        if label in label_to_entities_full and len(label_to_entities_full[label]) > 1
    ]

    # Limit the number of swaps to the available swappable labels
    n = min(n, len(swappable_labels))

    # Randomly select N labels for swapping
    selected_labels = random.sample(swappable_labels, n)

    # Perform swaps
    for label in selected_labels:
        original_entities = label_to_entities[label]
        full_entities = label_to_entities_full[label]

        for original_entity in original_entities:
            if len(full_entities) >= 2:
                # Select a replacement entity randomly, ensuring it's not the same as the original
                replacement_entity = random.choice(
                    [e for e in full_entities if e != original_entity]
                )

                # Swap the original entity with the replacement in the text
                text = text.replace(original_entity, replacement_entity)

    return text



class RemoveRandomWordsConfig(BaseModel):
    enabled: bool = Field(..., description="Whether random word removal is enabled.")
    number: int = Field(0, description="Number of random words to remove.")


class TokenizeConfig(BaseModel):
    should_shuffle: bool = False
    should_remove_sentence: bool = False
    should_remove_random_words: RemoveRandomWordsConfig = Field(
        default_factory=lambda: RemoveRandomWordsConfig(enabled=False, number=0),
        description="Configuration for removing random words.",
    )
    should_replace_words_with_lemma_form: RemoveRandomWordsConfig = Field(
        default_factory=lambda: RemoveRandomWordsConfig(enabled=False, number=0),
        description="Configuration for replacing words with lemmas.",
    )
    should_replace_conjunctions: RemoveRandomWordsConfig = Field(
        default_factory=lambda: RemoveRandomWordsConfig(enabled=False, number=0),
        description="Configuration for replacing conjunctions with anothers.",
    )
    should_replace_ner: RemoveRandomWordsConfig = Field(
        default_factory=lambda: RemoveRandomWordsConfig(enabled=False, number=0),
        description="Configuration for replacing NER entities by others of the same label.",
    )
    should_random_sentence_from_another_paper: RemoveRandomWordsConfig = Field(
        default_factory=lambda: RemoveRandomWordsConfig(enabled=False, number=0),
        description="Configuration for replacing NER entities by others of the same label.",
    )


class Criteria(Enum):
    Coherence = "Coherence"
    Fluency = "Fluency"
    Consistency = "Consistency"
    Relevance = "Relevance"


def get_config(language: str) -> Tuple[TokenizeConfig, List[Criteria]]:
    # Select a random sample of namedtuple classes (instead of instances)
    selected_criteria_classes = random.sample(
        [
            Criteria.Coherence,
            Criteria.Fluency,
            Criteria.Consistency,
            Criteria.Relevance,
        ],
        k=random.randint(1, 4)
    )
    # Initialize the config dictionary
    config_dict = {
        "should_shuffle": False,
        "should_remove_sentence": False,
        "should_remove_random_words": {"enabled": False, "number": 0},
        "should_replace_words_with_lemma_form": {"enabled": False, "number": 0},
        "should_replace_conjunctions": {"enabled": False, "number": 0},
        "should_replace_ner": {"enabled": False, "number": 0},
        "should_random_sentence_from_another_paper": {"enabled": False, "number": 0},
    }

    # Update the config_dict based on selected criteria classes
    for criteria_class in selected_criteria_classes:
        if criteria_class == Criteria.Fluency:
            config_dict["should_replace_words_with_lemma_form"]["enabled"] = True if language not in ("chinese", "japanese", "yoruba") else False
            config_dict["should_replace_words_with_lemma_form"]["number"] = (random.randint(30, 50))
            config_dict["should_replace_conjunctions"]["enabled"] = True if language not in ("chinese", "japanese", "turkish") else False
            config_dict["should_replace_conjunctions"]["number"] = random.randint(2, 6)
            config_dict["should_remove_random_words"]["enabled"] = True
            config_dict["should_remove_random_words"]["number"] = random.randint(8, 15) if language not in ("chinese", "japanese")\
                else random.randint(12, 15)
        elif criteria_class == Criteria.Coherence:
            config_dict["should_shuffle"] = True
        elif criteria_class == Criteria.Consistency:
            config_dict["should_replace_ner"]["enabled"] = True
            config_dict["should_replace_ner"]["number"] = random.randint(3, 10)
        elif criteria_class == Criteria.Relevance:
            config_dict["should_random_sentence_from_another_paper"]["enabled"] = True
            config_dict["should_random_sentence_from_another_paper"]["number"] = (random.randint(1, 3))
            config_dict["should_remove_sentence"] = True

    # Create the TokenizeConfig object with the updated config_dict
    config = TokenizeConfig(
        should_shuffle=config_dict["should_shuffle"],
        should_replace_words_with_lemma_form=RemoveRandomWordsConfig(
            **config_dict["should_replace_words_with_lemma_form"]
        ),
        should_replace_conjunctions=RemoveRandomWordsConfig(
            **config_dict["should_replace_conjunctions"]
        ),
        should_remove_random_words=RemoveRandomWordsConfig(
            **config_dict["should_remove_random_words"]
        ),
        should_replace_ner=RemoveRandomWordsConfig(**config_dict["should_replace_ner"]),
        should_random_sentence_from_another_paper=RemoveRandomWordsConfig(
            **config_dict["should_random_sentence_from_another_paper"]
        ),
        should_remove_sentence=config_dict["should_remove_sentence"]
    )

    return config, selected_criteria_classes


def noise_text(
    text: str, config: TokenizeConfig, language: str, human_summary: str = ""
, full_article: str = "") -> str:
    tokenized_sentences = sent_tokenize_sentences(text, language)

    if config.should_shuffle:
        print("should_shuffle")
        tokenized_sentences = shuffle_pairs(tokenized_sentences)

    # if config.should_remove_sentence:
    #     tokenized_sentences.pop(random.randrange(len(tokenized_sentences)))

    if config.should_remove_random_words:
        print("should_remove_random_words")
        tokenized_sentences = remove_overlapping_words(
            tokenized_sentences,
            human_summary,
            config.should_remove_random_words.number,
            language,
        )
        tokenized_sentences = detokenize_two_levels(tokenized_sentences)
        tokenized_sentences = sent_tokenize_sentences(tokenized_sentences, language)

    if config.should_replace_words_with_lemma_form.enabled:
        print("replace_random_words_with_lemmas")
        words_tokenized = [
            word_tokenize_sentence(sent, language) for sent in tokenized_sentences
        ]
        tokenized_words = replace_random_words_with_lemmas(
            words_tokenized,
            config.should_replace_words_with_lemma_form.number,
            language,
        )
        text = detokenize_two_levels(tokenized_words)
        tokenized_sentences = sent_tokenize_sentences(text, language)

    if config.should_replace_conjunctions.enabled:
        print("should_replace_conjunctions")
        text = detokenize_one_level(tokenized_sentences)
        converted_text = replace_conjunctions(
            text, language, num_replacements=config.should_replace_conjunctions.number
        )
        tokenized_sentences = sent_tokenize_sentences(converted_text, language)

    if config.should_replace_ner.enabled:
        print("should_replace_ner")
        detokenized_text = detokenize_one_level(tokenized_sentences)
        ner_results_in_full_article = LANGUAGE_TO_NER[language](full_article)
        ner_results_in_summary = LANGUAGE_TO_NER[language](detokenized_text)

        print(f"ner_results_in_summary: {ner_results_in_summary}")
        print(f"ner_results_in_full_article: {ner_results_in_full_article}")
        print(f"detokenized_text: {detokenized_text}")
        print(f"n: {config.should_replace_ner.number}")
        text = swap_entities_in_text(
            detokenized_text, ner_results_in_summary, ner_results_in_full_article, n=config.should_replace_ner.number
        )
        tokenized_sentences = sent_tokenize_sentences(text, language)

    if config.should_random_sentence_from_another_paper.enabled:
        print("should_random_sentence_from_another_paper")
        tokenized_sentences = insert_random_sentence(
            tokenized_sentences,
            language,
            config.should_random_sentence_from_another_paper.number,
        )

        text = detokenize_one_level(tokenized_sentences)
    return text
