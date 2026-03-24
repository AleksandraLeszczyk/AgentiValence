import logging
import os

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from piai.literature_sage import literature_sage
from piai.utils import parse_event_to_markdown

logger = logging.getLogger()
load_dotenv(override=True)


PI_PROMPT = """You are a principal investigator in a research project in area of quantum chemistry. 
You are a critical thinker who has an eye for detail and do not tolerate errors.
You enjoy exploring new ideas but you always stick to facts and always inform if any thought is not supported by external source.
You finish your analysis with conclusion. Don't suggest further investigation - just do it without user's prompt.
You coordinate a group of experts.
You can assign tasks for a group of experts using tools. This is a list of experts:

literature_sage - has an access to knowledge base. Finds relevant publications, and based on his findings, writes a summary of past research and suggest computation methodology for a current research task.
code_mage - specialized in writing scripts that run calculations. Has ability to run calculations.
calculation_analyst - interprets output of calculations. Can instruct calculation_runner to run calculations again with changes if there is a problem. Writes summary and conclusions of the experiment of the output looks correctly.
writer - converges past research and calculation results into a paper, write conclusions, introduction, and abstract.

With this board of experts, your job include:
Create plan for research project. If you are not sure if your idea for a task is feasible, you can ask any expert if he can perform it and if he has any suggestions.
Assign tasks for experts. Review results of each task before you do next step. If the results are not satisfying, you modify task and assign a task once again with modified requirements.
Never answer with question.
Good luck! Here is a research project you are assigned with:
"""


@tool
def ask_literature_sage(query: str) -> dict:
    """Search for information."""
    logger.info("Asking Literature Sage: %s" % query)
    return literature_sage.invoke(
        {"messages": [HumanMessage(query)]}
    )["messages"][-1].content


@tool
def ask_code_mage(query: str) -> str:
    """Search for information."""
    logger.info("Asking Code Mage: %s" % query)
    raise NotImplementedError


@tool
def ask_calculation_analyst(query: str) -> str:
    """Search for information."""
    logger.info("Asking Calculation Analyst: %s" % query)
    raise NotImplementedError


@tool
def ask_writer(query: str) -> str:
    """Search for information."""
    logger.info("Asking Writer: %s" % query)
    raise NotImplementedError


model_principal_investigator = ChatOpenAI(
    model=os.environ["MODEL_PRINCIPAL_INVESTIGATOR"]
)
principal_investigator = create_agent(
    model_principal_investigator,
    tools=[
        ask_literature_sage,
        # ask_calculation_analyst, ask_calculation_runner, ask_writer
    ],
    system_prompt=SystemMessage(
        content=[
            {
                "type": "text",
                "text": PI_PROMPT,
            }
        ]
    ),
)


def format_context(context: list[str | Document]):
    result = "<h2 style='color: #ff7800;'>Relevant Context</h2>\n\n"
    for doc in context:
        result += (
            f"<span style='color: #ff7800;'>Source: {doc.metadata['source']}</span>\n\n"
        )
        result += doc.page_content + "\n\n"
    return result


def chat_with_principal_investigator(history: list[dict]):
    """
    Generator function that streams agent events to the Gradio UI.
    Yields: (updated_history, event_markdown_string)
    """
    last_message_content = history[-1]["content"]
    
    if isinstance(last_message_content, list):
        last_text = last_message_content[0]["text"]
    else:
        last_text = last_message_content

    # Initialize accumulation strings
    event_accumulator = "### 🔬 Research Progress\n"
    full_answer = ""

    # We use .stream to get chunks of the agent's execution
    # Ensure your 'principal_investigator' agent supports streaming
    for chunk in principal_investigator.stream(
        {"messages": [HumanMessage(content=last_text)]},
        stream_mode="values" # or "updates" depending on your Graph configuration
    ):
        # 1. Update the Event Log (Right Panel)
        # We parse the chunk to see what the agent is currently doing
        new_event_text = parse_event_to_markdown(chunk["messages"][-1])
        event_accumulator += f"\n{new_event_text}"
        
        # 2. Update the Final Answer (if available in the current chunk)
        if "messages" in chunk:
            last_msg = chunk["messages"][-1]
            if isinstance(last_msg, AIMessage):
                full_answer = last_msg.content

        # Yield current state to the UI to create the "real-time" effect
        # We temporarily append the partial answer to a copy of history
        temp_history = history + [{"role": "assistant", "content": full_answer}]
        yield temp_history, event_accumulator

    # Final yield to ensure everything is locked in
    history.append({"role": "assistant", "content": full_answer})
    yield history, event_accumulator