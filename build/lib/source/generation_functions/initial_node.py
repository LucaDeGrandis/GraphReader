from ..llm.llm import LLM


SYSTEM_STARTING_NODE_SELECTION_TEMPLATE = """As an intelligent assistant, your primary objective is to answer questions based on information contained within a text. To facilitate this objective, a graph has been created from the text, comprising the following elements:
1. Text Chunks: Chunks of the original text.
2. Atomic Facts: Smallest, indivisible truths extracted from text chunks.
3. Nodes: Key elements in the text (noun, verb, or adjective) that correlate with several atomic facts derived from different text chunks.

Your current task is to check a list of nodes, with the objective of selecting the most relevant initial nodes from the graph to efficiently answer the question. You are given the question, the rational plan, and a list of node key elements. These initial nodes are crucial because they are the starting point for searching for relevant information.

Requirements:
#####
1. Once you have selected a starting node, assess its relevance to the potential answer by assigning a score between 0 and 100. A score of 100 implies a high likelihood of relevance to the answer, whereas a score of 0 suggests minimal relevance.
2. Present each chosen starting node in a separate line, accompanied by its relevance score. Format each line as follows: Node: [Key Element of Node], Score: [Relevance Score].
3. Please select at least 3 starting nodes, ensuring they are non-repetitive and diverse.
4. In the user’s input, each line constitutes a node. When selecting the starting node, please make your choice from those provided, and refrain from fabricating your own. The nodes you output must correspond exactly to the nodes given by the user, with identical wording.
5. Format the output as a list of dictionaries with keys "node_name" and "node_score".
#####

Example:
#####
User:

Question: <QUESTION>

Rational Plan: <RATIONAL PLAN>

Nodes:
- <NODE_NAME>: <KEY_ELEMENTS>
- <NODE_NAME>: <KEY_ELEMENTS>
...

Assistant: [
    {"node_name": <NODE_NAME>, "node_score": <NODE_SCORE>},
    {"node_name": <NODE_NAME>, "node_score": <NODE_SCORE>},
    {"node_name": <NODE_NAME>, "node_score": <NODE_SCORE>},
    ...
]
#####

Finally, I emphasize again that you need to select the starting node from the given Nodes, and it must be consistent with the words of the node you selected. Please strictly follow the above format. Let’s begin."""


USER_STARTING_NODE_SELECTION_TEMPLATE = """Question: {question}

Rational Plan: {plan}

Nodes:
{nodes}"""


NODES_TEMPLATE = """- {node_name}: {node_key_elements}\n"""


def starting_node_selection_generation(
    llm: LLM,
    **kwargs,
) -> str:
    nodes_str = ""
    for node in kwargs['nodes']:
        nodes_str += NODES_TEMPLATE.format(**{
            'node_name': node['node_name'],
            'node_key_elements': ', '.join(node['node_key_elements']),
        })
    nodes_str = nodes_str.strip()

    messages_list = [
        {'role': 'system', 'content': SYSTEM_STARTING_NODE_SELECTION_TEMPLATE},
        {'role': 'user', 'content': USER_STARTING_NODE_SELECTION_TEMPLATE.format(**{
            'question': kwargs['question'],
            'plan': kwargs['plan'],
            'nodes': nodes_str,
        })},
    ]

    messages = llm.prepare_messages(messages_list)
    gen = llm.generate(messages)

    new_notebook = ""
    new_action = ""

    return gen, new_notebook, new_action
