from ..llm.llm import LLM


SYSTEM_ATOMIC_FACTS_EXPLORATION_TEMPLATE = """As an intelligent assistant, your primary objective is to answer questions based on information contained within a text. To facilitate this objective, a graph has been created from the text, comprising the following elements:
1. Text Chunks: Chunks of the original text.
2. Atomic Facts: Smallest, indivisible truths extracted from text chunks.
3. Nodes: Key elements in the text (noun, verb, or adjective) that correlate with several atomic facts derived from different text chunks.

Your current task is to check a node and its associated atomic facts, with the objective of determining whether to proceed with reviewing the text chunk corresponding to these atomic facts.
Given the question, the rational plan, previous actions, notebook content, and the current node’s atomic facts and their corresponding chunk IDs, you have the following Action Options:
#####
1. read_chunk(List[ID]): Choose this action if you believe that a text chunk linked to an atomic fact may hold the necessary information to answer the question. This will allow you to access more complete and detailed information.
2. stop_and_read_neighbor(): Choose this action if you ascertain that all text chunks lack valuable information.
#####

Strategy:
#####
1. Reflect on previous actions and prevent redundant revisiting nodes or chunks.
2. You can choose to read multiple text chunks at the same time.
3. Atomic facts only cover part of the information in the text chunk, so even if you feel that the atomic facts are slightly relevant to the question, please try to read the text chunk to get more complete information.
#####

Do the following:
1. First, combine your current notebook with new insights and findings about the question from current atomic facts, creating a more complete version of the notebook that contains more valid information.
2. Based on the given question, the rational plan, previous actions, and notebook content, analyze how to choose the next action.
3. Choose the next action from the Action Options mentioned above and provide the function call with the actual parameter.
The next action must be selected among read_chunk(List[ID]) and stop_and_read_neighbor() (Here is the Action you selected from Action Options, which is in the form of a function call as mentioned before. The formal parameter in parentheses should be replaced with the actual parameter.)

Response format:
#####
*Updated Notebook* (delimited by ```):
```<UPDATED_NOTEBOOK>```

*Rationale for Next Action*:
<RATIONALE>

*Chosen Action*:
<NEXT_ACTION>
#####

Finally, it is emphasized again that even if the atomic fact is only slightly relevant to the question, you should still look at the text chunk to avoid missing information. You should only choose stop_and_read_neighbor() when you are very sure that the given text chunk is irrelevant to the question. Please strictly follow the above format. Let’s begin."""


USER_ATOMIC_FACTS_EXPLORATION_TEMPLATE = """Question: {question}

Rational Plan: {plan}

Previous Actions: {previous_actions}

Notebook Content (delimited by ```): ```{notebook_content}```

Atomic Facts: {atomic_facts}"""


def read_node_atomic_facts_generation(
    llm: LLM,
    **kwargs,
) -> str:
    atomic_facts_str = '\n- '.join(kwargs['atomic_facts']).strip()

    messages_list = [
        {'role': 'system', 'content': SYSTEM_ATOMIC_FACTS_EXPLORATION_TEMPLATE},
        {'role': 'user', 'content': USER_ATOMIC_FACTS_EXPLORATION_TEMPLATE.format(**{
            'question': kwargs['question'],
            'plan': kwargs['plan'],
            'previous_actions': kwargs['previous_actions'],
            'notebook_content': kwargs['notebook_content'],
            'atomic_facts': atomic_facts_str,
        })},
    ]

    messages = llm.prepare_messages(messages_list)
    gen = llm.generate(messages)

    new_notebook = gen.split('*Updated Notebook* (delimited by ```):')[1].split('```')[1].split('```')[0].strip()
    new_action = gen.split('*Chosen Action*:')[1].strip()

    return gen, new_notebook, new_action
