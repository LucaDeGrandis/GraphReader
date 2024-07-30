from typing import List, Dict, Optional
from abc import ABC, abstractmethod


MODEL_NAMES = {
    'gpt-4o': {
        'provider': 'openai',
        'model': 'gpt-4o-2024-05-13',
    },
}


class LLM(ABC):
    @abstractmethod
    def __init__(
        self,
        model_name: str,
        model_key: Optional[str] = None,
    ) -> None:
        """Initializes the LLM class with the LLM itself"""

        assert model_name in MODEL_NAMES, f'{model_name} is not a recognized model name.'

        model_kwargs = MODEL_NAMES[model_name]
        model_kwargs['key'] = model_key if model_key is not None else ''
        self.model_name = model_name
        self.prepare_llm(model_kwargs)

    @abstractmethod
    def prepare_llm(
        self,
        kwargs,
    ) -> None:
        pass

    @abstractmethod
    def prepare_messages(
        self,
        message_list: List[str],
    ):
        pass

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
    ):
        pass
