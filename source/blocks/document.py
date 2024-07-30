from typing import Dict, Any
from ..document_functions.dom_heuristics import recursive_text_splitted
import nltk
nltk.download('punkt')


class Document:
    def __init__(
        self,
        document_graph: Dict[str, Any]
    ) -> None:
        """
        *args*
            *document_graph*: the document DOM.
        """

        self.document_graph = recursive_text_splitted([document_graph])[0]
