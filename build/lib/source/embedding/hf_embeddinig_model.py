from sentence_transformers import SentenceTransformer
from .embedding_model import Embedding_Model


class HF_Embedding_Model(Embedding_Model):
    def __init__(
        self,
        model_name: str,
        model_key: str = None,
    ) -> None:
        """Initializes the LLM class with the LLM itself"""

        super().__init__(model_name, model_key)

    def prepare_embedding_model(
        self,
        kwargs,
    ) -> None:
        self.model = SentenceTransformer(kwargs['model'])

    def __call__(
        self,
        text_list,
    ):
        return self.model.encode(text_list)
