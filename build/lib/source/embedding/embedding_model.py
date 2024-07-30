from abc import ABC, abstractmethod


EMBEDDING_MODEL_NAMES = {
    'phrase-bert': {
        'provider': 'huggingface',
        'model': 'whaleloops/phrase-bert',
    },
}


class Embedding_Model(ABC):
    @abstractmethod
    def __init__(
        self,
        model_name: str,
        model_key: str = None,
    ) -> None:
        """Initializes the LLM class with the LLM itself"""

        assert model_name in EMBEDDING_MODEL_NAMES, f'{model_name} is not a recognized model name.'

        model_kwargs = EMBEDDING_MODEL_NAMES[model_name]
        model_kwargs['key'] = model_key
        self.model_name = model_name
        self.prepare_embedding_model(model_kwargs)

    @abstractmethod
    def prepare_embedding_model(
        self,
        kwargs,
    ) -> None:
        pass

    @abstractmethod
    def __call__(
        self,
        text_list,
    ):
        pass
