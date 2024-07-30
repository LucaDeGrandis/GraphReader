from typing import Any, Dict
from ..blocks.document import Document
from ..llm.llm import LLM
from ..document_functions.dom_heuristics import extract_elements_from_dom
from ..generation_functions.rational_plan import rational_plan_generation
from ..generation_functions.initial_node import starting_node_selection_generation
from ..generation_functions.atomic_facts import read_node_atomic_facts_generation
from ..generation_functions.chunk_exploration import read_node_chunk_generation


def answer_user_query(
    llm: LLM,
    user_query: str,
    document: Document,
) -> Any:
    # Initialization
    generations = []
    queue = []
    notebook = ""
    notebooks = []
    actions = []
    rational_plan = ""
    initial_nodes = []

    # Create the queue
    queue.append(
        (rational_plan_generation, {}),
        (starting_node_selection_generation, {}),
    )

    # Iterate
    while queue:
        kwargs = {
            'user_query': user_query,
            'notebook': notebook,
            'actions': ', '.join(actions),
            'rational_plan': rational_plan,
        }
        current_action, additional_kwargs = queue.pop(0)
        for key, item in additional_kwargs.items():
            kwargs[key] = item
        gen, new_notebook, new_action, kwout = current_action(llm, **kwargs)

        generations.append(gen)
        notebook = new_notebook
        notebooks.append(new_notebook)
        actions.append(new_action)

        # Optional objects
        if 'rational_plan' in kwout:
            rational_plan = kwout['rational_plan']
        if 'initial_nodes' in kwout:
            initial_nodes = kwout['initial_nodes']

        # Transform the new action into a function
        if new_action:
            if ('read_chunk' in new_action) or (
                ('termination' in new_action) and ('remaining_chunks' in kwargs) and (kwargs['remaining_chunks'])
            ):
                if ('remaining_chunks' not in kwargs) or ('remaining_chunks' in kwargs and not kwargs['remaining_chunks']):
                    chunk_ids = list(map(int, new_action.split('([')[1].split('])')[0].split(',')))
                elif ('remaining_chunks' in kwargs and kwargs['remaining_chunks']):
                    chunk_ids = kwargs['remaining_chunks']
                chunk_id = chunk_ids.pop(0)
                chunk = prepare_ith_chunk(kwargs['section'], chunk_id)
                additional_kwargs = {
                    'chunk_id': chunk_id,
                    'chunk': chunk,
                }
                queue.append((read_node_chunk_generation, additional_kwargs))

            elif ('search_more' in new_action) and ('search_more' not in actions[-1]):
                additional_kwargs = {
                    'chunk_id': kwargs['chunk_id'],
                    'chunk': kwargs['chunk'],
                }
                queue.append((read_node_chunk_generation, additional_kwargs))

            elif ('read_previous_chunk' in new_action) or ('read_subsequent_chunk' in new_action):
                if 'previous' in new_action:
                    new_chunk_id = kwargs['chunk_id'] - 1
                else:
                    new_chunk_id = kwargs['chunk_id'] + 1
                chunk = prepare_ith_chunk(kwargs['section'], new_chunk_id)
                if chunk is None:
                    continue
                additional_kwargs = {
                    'chunk_id': new_chunk_id,
                    'chunk': chunk,
                }
                queue.append((read_node_chunk_generation, additional_kwargs))

            elif ('stop_and_read_neighbor' in new_action):
                pass #UNFNISHED

        # Move to new node
        if not queue:  # No queue means that
            if initial_nodes:
                target_node = initial_nodes.pop(0)
                target_section = prepare_ith_node(target_node['node_name'], document)
                atomic_facts = prepare_ith_node_atomic_facts(target_section)
                additional_kwargs = {
                    'node_name': target_node['node_name'],
                    'atomic_facts': atomic_facts,
                    'section': target_section,
                }
                queue.append((read_node_atomic_facts_generation, additional_kwargs))


def prepare_ith_node(
    node_name: str,
    document: Document,
) -> Dict[str, Any]:
    sections = extract_elements_from_dom([document.document_graph], 'Section')
    target_section = list(filter(lambda x: x['text'] == node_name, sections))[0]

    return target_section


def prepare_ith_node_atomic_facts(
    section: Dict[str, Any],
):
    atomic_facts = []
    for _el in section['children']:
        if _el['atomic_facts']:
            atomic_facts.append(_el['atomic_facts'])


def prepare_ith_chunk(
    section: Dict[str, Any],
    chunk_id: int,
):
    section_chunks = extract_elements_from_dom(section['children'], 'Chunk')
    if (chunk_id >= len(section_chunks)) or (chunk_id < 0):
        return None
    else:
        return section_chunks[chunk_id]


def prepare_neighbours_for_exploration(
    section: Dict[str, Any],
    document: Document,
):
    target_sections = section['neighbours']
    sections = extract_elements_from_dom([document.document_graph], 'Section')
    candidate_sections = list(filter(lambda x: x['text'] in target_sections, sections))

    candidate_sections_keypoints = []
    for _section in candidate_sections:
        candidate_sections_keypoints.append({
            'node_name': _section['text'],
            'node_key_elements': _section['key_points'],
        })

    return candidate_sections_keypoints
