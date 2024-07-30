from typing import List, Dict, Union, Optional
from .llm import LLM
from openai import OpenAI


class Openai_LLM(LLM):
    def __init__(
        self,
        model_name: str,
        model_key: Optional[str],
    ):
        super().__init__(model_name, model_key)

    def prepare_llm(
        self,
        kwargs,
    ):
        self.client = OpenAI(api_key=kwargs['key'])

    def prepare_messages(
        self,
        messages_list: List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]],
    ) -> List[Dict[str, Union[str, List[Dict[str, Union[str, Dict[str, str]]]]]]]:
        messages = []

        system_message = list(filter(lambda x: x['role'] == 'system', messages_list))
        assert len(system_message) <= 1
        if len(system_message) == 1:
            messages.append(system_message[0])

        other_messages = list(filter(lambda x: x['role'] != 'system', messages_list))
        for _message in other_messages:
            assert _message['role'] != messages[-1]['role']
            if _message['role'] == 'user':
                messages.append({
                    'role': 'user',
                    'content': [{
                        'type': 'text',
                        'text': _message['content']
                    }]
                })
            if _message['role'] == 'ai':
                messages.append({
                    'role': 'ai',
                    'content': [{
                        'type': 'text',
                        'text': _message['content']
                    }]
                })

        return messages

    def generate(
        self,
        messages: List[Dict[str, str]],
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
        )

        return response.choices[0].message.content
