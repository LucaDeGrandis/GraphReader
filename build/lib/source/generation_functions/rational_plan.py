from ..llm.llm import LLM


RATIONAL_PLAN_SYSTEM_TEMPLATE = """As an intelligent assistant, your primary objective is to answer the question by gathering supporting facts from a given document. To facilitate this objective, the first step is to make a rational plan based on the question. This plan should outline the step-by-step process to resolve the question and specify the key information required to formulate a comprehensive answer.

Example:
#####
User: Who had a longer tennis career, Danny or Alice?

Assistant: In order to answer this question, we first need to find the length of Danny’s and Alice’s tennis careers, such as the start and retirement of their careers, and then compare the two.
#####

Please strictly follow the above format. Let’s begin."""


KEY_POINTS_USER_TEMPLATE = """Here is the long text (delimited by ```): ```{text}```."""


def rational_plan_generation(
    llm: LLM,
    user_instruction: str
) -> str:
    """Uses the LLM for the generation of a rational plan."""

    messages_list = [
        {'role': 'system', 'content': RATIONAL_PLAN_SYSTEM_TEMPLATE},
        {'role': 'user', 'content': user_instruction},
    ]

    messages = llm.prepare_messages(messages_list)

    response = llm.generate(messages)

    return response
