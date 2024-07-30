from typing import List
import re
from nltk.stem import PorterStemmer


def remove_number_point_prefix(
    text: str
) -> str:
    """
    Removes the number point prefix from the given text.

    Args:
        text (str): The input text.

    Returns:
        str: The text with the number point prefix removed.

    Examples:
        >>> remove_number_point_prefix('1. Hello World')
        'Hello World'
        >>> remove_number_point_prefix('2. Python is awesome')
        'Python is awesome'
    """
    return re.sub(r'^\d+\.', '', text).strip()


def stem_words(
    words: List[str]
) -> List[str]:
    """
    Stems a list of words using the Porter stemming algorithm.

    Args:
        words (list): A list of words to be stemmed.

    Returns:
        list: A list of stemmed words.

    Example:
        >>> words = ['running', 'jumps', 'jumping']
        >>> stem_words(words)
        ['run', 'jump', 'jump']
    """
    porter = PorterStemmer()
    stemmed_words = [porter.stem(word) for word in words]
    return stemmed_words


def word_normalization(
    tags: List[str],
) -> List[str]:
    """
    Normalizes a list of tags by converting them to lowercase, removing punctuation,
    stemming the words, and stripping leading and trailing whitespaces.

    Args:
        tags (List[str]): The list of tags to be normalized.

    Returns:
        List[str]: The normalized list of tags.
    """
    tags = [x.lower() for x in tags]
    tags = [re.sub(r'[^\w\s]', ' ', x) for x in tags]
    tags = stem_words(tags)
    tags = [x.strip() for x in tags]

    return tags
