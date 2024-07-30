from typing import Any, List
import nltk
from nltk.tokenize import sent_tokenize


nltk.download('punkt')


def extract_elements_from_dom(
    dom: Any,
    obj_type: str,
):
    """
    Extracts elements of a specific object type from a DOM.

    Args:
        dom (Any): The DOM to extract elements from.
        obj_type (str): The object type to extract.

    Returns:
        List: A list of elements of the specified object type found in the DOM.
    """
    res = []
    for _el in dom:
        if ('obj_type' in _el.keys()) and (_el['obj_type'] == obj_type):
            res.append(_el)
        elif 'children' in _el:
            candidates = extract_elements_from_dom(_el['children'], obj_type)
            res.extend(candidates)

    return res


def split_text(
    text: str,
    n_tokens: int,
) -> List[str]:
    """
    Splits the given text into chunks of sentences, where each chunk contains
    approximately `n_tokens` number of tokens.

    Args:
        text (str): The input text to be split.
        n_tokens (int): The desired number of tokens in each chunk.

    Returns:
        List[str]: A list of chunks, where each chunk is a string containing
        a group of sentences.

    Example:
        >>> text = "This is a sample text. It contains multiple sentences."
        >>> n_tokens = 10
        >>> split_text(text, n_tokens)
        ['This is a sample text.', 'It contains multiple sentences.']
    """
    sentences = sent_tokenize(text)
    result = []
    current_tokens = 0
    current_sentence = ""

    for sentence in sentences:
        current_sentence += ' ' + sentence
        n = len(nltk.word_tokenize(sentence))

        current_tokens += n
        if current_tokens > n_tokens:
            result.append(current_sentence)
            current_sentence = ""
            current_tokens = 0

    return result


def recursive_text_splitted(
    dom: List[Any],
    n_tokens: int,
) -> Any:
    """
    Recursively splits the text in the DOM elements into chunks of specified number of tokens.

    Args:
        dom (List[Any]): The DOM elements to process.
        n_tokens (int): The number of tokens to split the text into.

    Returns:
        Any: The modified DOM elements with the text split into chunks.

    """
    res = []
    for _el in dom:
        if 'text' in _el.keys():
            _el['chunks'] = split_text(_el['text'], n_tokens)
            if 'children' in _el:
                _el['children'] = recursive_text_splitted(_el['children'], n_tokens)
        res.append(_el)
    return res
