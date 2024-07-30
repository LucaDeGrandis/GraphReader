from ..llm.llm import LLM


SYSTEM_READ_CHUNK_TEMPLATE = """As an intelligent assistant, your primary objective is to answer questions based on information within a text. To facilitate this objective, a graph has been created from the text, comprising the following elements:
1. Text Chunks: Segments of the original text.
2. Atomic Facts: Smallest, indivisible truths extracted from text chunks.
3. Nodes: Key elements in the text (noun, verb, or adjective) that correlate with several atomic facts derived from different text chunks.

Your current task is to assess a specific text chunk and determine whether the available information suffices to answer the question. Given the question, rational plan, previous actions, notebook content, and the current text chunk, you have the following Action Options:
#####
1. search_more(): Choose this action if you think that the essential information necessary to answer the question is still lacking.
2. read_previous_chunk(): Choose this action if you feel that the previous text chunk contains valuable information for answering the question.
3. read_subsequent_chunk(): Choose this action if you feel that the subsequent text chunk contains valuable information for answering the question.
4. termination(): Choose this action if you believe that the information you have currently obtained is enough to answer the question. This will allow you to summarize the gathered information and provide a  final answer.

Strategy:
#####
1. Reflect on previous actions and prevent redundant revisiting of nodes or chunks.
2. You can only choose one action.
#####

Do the following:
1. First, combine your previous notes with new insights and findings about the question from current text chunks, creating a more complete version of the notebook that contains more valid information.
2. Based on the given question, rational plan, previous actions, and notebook content, analyze how to choose the next action.
3. Choose the next action from the Action Options mentioned above and provide the function call with the actual parameter.
The next action must be selected among search_more(), read_previous_chunk(), read_subsequent_chunk(), and termination() (Here is the Action you selected from Action Options, which is in the form of a function call as mentioned before. The formal parameter in parentheses should be replaced with the actual parameter.)

Response format:
#####
*Updated Notebook* (delimited by ```):
```<UPDATED_NOTEBOOK>```

*Rationale for Next Action*:
<RATIONALE>

*Chosen Action*:
<NEXT_ACTION>
#####

Please strictly follow the above format. Let’s begin."""


USER_READ_CHUNK_TEMPLATE = """Question: {question}

Rational Plan: {plan}

Previous Actions: {previous_actions}

Notebook Content (delimited by ```): ```{notebook_content}```

Current Text Chunk: {chunk}"""


def read_node_chunk_generation(
    llm: LLM,
    **kwargs,
) -> str:
    messages_list = [
        {'role': 'system', 'content': SYSTEM_READ_CHUNK_TEMPLATE},
        {'role': 'user', 'content': USER_READ_CHUNK_TEMPLATE.format(**{
            'question': kwargs['question'],
            'plan': kwargs['plan'],
            'previous_actions': kwargs['previous_actions'],
            'notebook_content': kwargs['notebook_content'],
            'chunk': kwargs['chunk'],
        })},
    ]

    messages = llm.prepare_messages(messages_list)
    gen = llm.generate(messages)

    new_notebook = gen.split('*Updated Notebook* (delimited by ```):')[1].split('```')[1].split('```')[0].strip()
    new_action = gen.split('*Chosen Action*:')[1].strip()

    return gen, new_notebook, new_action
