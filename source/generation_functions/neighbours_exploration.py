from ..llm.llm import LLM


SYSTEM_READ_NEIGHBOURS_TEMPLATE = """As an intelligent assistant, your primary objective is to answer questions based on information within a text. To facilitate this objective, a graph has been created from the text, comprising the following elements:
1. Text Chunks: Segments of the original text.
2. Atomic Facts: Smallest, indivisible truths extracted from text chunks.
3. Nodes: Key elements in the text (noun, verb, or adjective) that correlate with several atomic facts derived from different text chunks.

Your current task is to assess all neighboring nodes of the current node, with the objective of determining whether to proceed to the next neighboring node. Given the question, rational plan, previous actions, notebook content, and the neighbors of the current node, you have the following Action Options:
#####
1. read_neighbor_node(key element of node): Choose this action if you believe that any of the neighboring nodes may contain information relevant to the question. Note that you should focus on one neighbor node at a time.
2. termination(): Choose this action if you believe that none of the neighboring nodes possess information that could answer the question.
#####

Strategy:
#####
1. Reflect on previous actions and prevent redundant revisiting of nodes or chunks.
2. You can only choose one action. This means that you can choose to read only one neighbor node or choose to terminate.
#####

Do the following:
1. Based on the given question, rational plan, previous actions, and notebook content, analyze how to choose the next action.
2. Choose the next action from the Action Options mentioned above and provide the function call with the actual parameter.
The next action must be selected among read_neighbor_node(neighbor_node) and termination() (Here is the Action you selected from Action Options, which is in the form of a function call as mentioned before. The formal parameter in parentheses should be replaced with the actual parameter.)

Input Example:
#####
User:

Question: <QUESTION>

Rational Plan: <RATIONAL_PLAN>

Previous Actions: <PREVIOUS_ACTIONS>

Notebook Content (delimited by ```): ```<NOTEBOOK_CONTENT>```

Neighbouring Nodes:
- <NODE_NAME>: <KEY_ELEMENTS>
- <NODE_NAME>: <KEY_ELEMENTS>
...
####

Output Format:
####
*Rationale for Next Action*:
<RATIONALE>

*Chosen Action*:
<NEXT_ACTION>
#####

Please strictly follow the above format. Let’s begin."""


USER_READ_NEIGHBOURS_TEMPLATE = """Question: {question}

Rational Plan: {plan}

Previous Actions: {previous_actions}

Notebook Content (delimited by ```): ```{notebook_content}```

Neighbouring Nodes:
{nodes}"""


NODES_TEMPLATE = """- {node_name}: {node_key_elements}\n"""


def read_neighbours_generation(
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
        {'role': 'system', 'content': SYSTEM_READ_NEIGHBOURS_TEMPLATE},
        {'role': 'user', 'content': USER_READ_NEIGHBOURS_TEMPLATE.format(**{
            'question': kwargs['question'],
            'plan': kwargs['plan'],
            'previous_actions': kwargs['previous_actions'],
            'notebook_content': kwargs['notebook_content'],
            'nodes': nodes_str,
        })},
    ]

    messages = llm.prepare_messages(messages_list)
    gen = llm.generate(messages)

    new_notebook = ""
    new_action = gen.split('*Chosen Action*:')[1].strip()

    return gen, new_notebook, new_action
