from ..llm.llm import LLM
from ..document_functions.text_heuristics import remove_number_point_prefix


KEY_POINTS_SYSTEM_TEMPLATE = """You are now an intelligent assistant tasked with meticulously extracting both key elements and atomic facts from a long text.
1. Key Elements: The essential nouns (e.g., characters, times, events, places, numbers), verbs (e.g., actions), and adjectives (e.g., states, feelings) that are pivotal to the text narrative.
2. Atomic Facts: The smallest, indivisible facts, presented as concise sentences. These include propositions, theories, existences, concepts, and implicit elements like logic, causality, event sequences, interpersonal relationships, timelines, etc

Requirements:
#####
1. Ensure that all identified key elements are reflected within the corresponding atomic facts.
2. You should extract key elements and atomic facts comprehensively, especially those that are important and potentially query-worthy and do not leave out details.
3. Whenever applicable, replace pronouns with their specific noun counterparts (e.g., change I, He, She to actual names).
4. Ensure that the key elements and atomic facts you extract are presented in the same language as the original text (e.g., English or Chinese).
5. You should output a total of key elements and atomic facts that do not exceed 1024 tokens.
6. Your answer format for each line should be: [Serial Number], [Atomic Facts], [List of Key Elements, separated with ‘|’]
#####

Example:
#####
User:
One day, a father and his little son ......

Assistant:
1. One day, a father and his little son were going home. | father | little son | going home
2. ......
#####

Please strictly follow the above format. Let’s begin."""


KEY_POINTS_USER_TEMPLATE = """Here is the long text (delimited by ```): ```{text}```."""


def key_point_generation(
    llm: LLM,
    user_instruction: str
) -> str:
    messages_list = [
        {'role': 'system', 'content': KEY_POINTS_SYSTEM_TEMPLATE},
        {'role': 'user', 'content': KEY_POINTS_USER_TEMPLATE.format(**{
            'text': user_instruction,
        })},
    ]

    messages = llm.prepare_messages(messages_list)
    response = llm.generate(messages)

    key_elements = []
    atomic_facts = []

    for line in response.split('\n'):
        elements = line.split('|')[1:]
        fact = remove_number_point_prefix(line.split('|')[0])

        key_elements.extend(elements)
        atomic_facts.append(fact)

    return key_elements, atomic_facts
