import re
import random

# Hebrew

hebrew_words = [
    "ו",
    "או",
    "אבל",
    "ולא",
    "גם",
    "כי",
    "אם",
    "כש",
    "ש",
    "למרות ש",
    "בגלל ש",
    "עד ש",
    "מאז ש",
    "גם וגם",
    "או",
    "כמו",
    "מאשר",
    "כפי ש",
    "כדי",
    "אומנם",
    "אבל",
    "בתחילת",
    "בסוף",
    "לדוגמה",
    "דהיינו",
    "כידוע",
    "בזכות",
    "בעקבות",
    "כמו כן",
    "במקרה של",
    "לעומת",
    "בעוד",
]

arabic_words = [
    "و",
    "أو",
    "لكن",
    "ولا",
    "أيضًا",
    "إلا",
    "لأن",
    "إذا",
    "عندما",
    "حتى",
    "بينما",
    "لذلك",
    "كما",
    "لأنَّ",
    "منذ",
    "فقط",
    "لما",
    "إلى أن",
    "في حين",
    "إذ",
    "حتى إذا",
    "سواء",
]

spanish_words = [
    "y",
    "o",
    "pero",
    "ni",
    "porque",
    "si",
    "cuando",
    "hasta",
    "mientras",
    "aunque",
    "ya que",
    "pues",
    "como",
    "para que",
    "por lo tanto",
    "sin embargo",
    "antes de que",
    "después de que",
    "a menos que",
    "en caso de que",
    "ya que",
    "como si",
]

indonesian_words = [
    "jadi",
    "dengan",
    "sampai",
    "baik",
    "selain itu",
    "bahwa",
    "sebab",
    "di",
    "tetapi",
    "seperti",
    "jika",
    "sehingga",
    "ketika",
    "dan juga",
    "untuk",
    "walaupun",
    "dengan demikian",
    "juga",
    "pada saat",
    "namun",
    "di mana",
    "sebagai",
    "jikalau",
    "supaya",
    "dan",
    "oleh karena itu",
    "seandainya",
    "pada",
    "meskipun begitu",
    "sambil",
    "meskipun",
    "karena",
    "sementara",
    "atau",
    "selagi",
    "bila",
    "agar",
    "selama",
]

ukrainian_words = [
    "і",
    "та",
    "що",
    "або",
    "навіть",
    "коли",
    "через",
    "але",
    "до",
    "так",
    "як",
    "щоб",
    "з",
    "із",
    "також",
    "хоча",
    "в",
    "для",
    "у",
    "чи",
    "і",
    "та",
    "або",
    "чи",
    "але",
    "проте",
    "однак",
    "тим не менш",
    "тому",
    "зокрема",
    "коли",
    "якщо",
    "хоча",
    "оскільки",
    "бо",
    "щоб",
    "поки",
    "наскільки",
    "де",
    "хоч",
    "кожного разу як",
    "як",
    "як",
    "не тільки але й",
    "або",
    "ні",
    "для того щоб",
    "аби",
    "адже",
    "тому що",
    "якщо б",
    "у разі",
    "все ж",
    "тим часом",
]

yoruba_words = [
    "abi",
    "amọ",
    "ati",
    "bi",
    "dẹ",
    "ko",
    "pe",
    "pẹlu",
    "si",
    "ṣugbọn",
    "tabi",
    "yala",
]

LANGUAGE_TO_LIST_OF_WORDS = {
    "arabic": arabic_words,
    "hebrew": hebrew_words,
    "yoruba": yoruba_words,
    "ukrainian": ukrainian_words,
    "spanish": spanish_words,
    "indonesian": indonesian_words,
}

# List of conjunctions (conductors)

# Function to replace conjunctions in a text
import re

# List of conjunctions (conductors)
conjunctions = [
    "abi",
    "amọ",
    "ati",
    "bi",
    "dẹ",
    "ko",
    "pe",
    "pẹlu",
    "si",
    "ṣugbọn",
    "tabi",
    "yala",
]


# Function to replace conjunctions in a text


def replace_conjunctions(text: str, language: str, num_replacements=None) -> str:
    """
    This function searches for words in the input text that match any word
    from the provided conjunctions list and replaces them with a random word
    from the same list, up to a specified number of replacements.

    Args:
    - text (str): The input text to search and replace conjunctions in.
    - conjunctions (list, optional): List of words to search and replace. If None, defaults to the provided list.
    - num_replacements (int, optional): The number of words to replace. If None, replaces all occurrences.

    Returns:
    - str: The modified text with conjunctions replaced.
    """
    # Default list of conjunctions if not provided

    conjunctions = LANGUAGE_TO_LIST_OF_WORDS[language]

    # Split the text into words
    words = text.split()

    # Counter for number of replacements made
    replacements_made = 0

    # Iterate over the words and replace conjunctions
    for i, word in enumerate(words):
        if word in conjunctions:
            print("before conjunction", word)
            if num_replacements is not None and replacements_made >= num_replacements:
                break  # Stop replacing if the max number of replacements is reached

            # Replace with a random word from the list (excluding the current word)
            replacement = random.choice([w for w in conjunctions if w != word])
            print("after conjunction", replacement)
            words[i] = replacement
            replacements_made += 1  # Increment the replacement counter

    # Join the words back into a string
    return " ".join(words)
