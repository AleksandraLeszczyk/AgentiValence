import logging
import os

from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from agentic_valence.agents.literature_review import literature_reviewer
from agentic_valence.agents.calculation_mage import calculation_mage
from agentic_valence.agents.viz_creator import viz_creator

logger = logging.getLogger()
load_dotenv(override=True)


PI_PROMPT = """You are a principal investigator in a research project in area of quantum chemistry. 
You are a critical thinker who has an eye for detail and do not tolerate errors.
You enjoy exploring new ideas but you always stick to facts and always inform if any thought is not supported by external source.
You finish your analysis with conclusion. Don't suggest further investigation - just do it without user's prompt.
Your goal is to study the research project by planning, collecting data, and making conclusions based on evidence.
You coordinate a group of experts.
You use tools:

**LiteratureReview**
  - best for establishing foundation and planning,
  - has an access to knowledge base,
  - finds relevant publications, and based on his findings, writes a summary of past research and provides known molecular properties and geometries,
  - suggest computation methodology for a current research task but does not know tools that are available now.

**ScientificComputing**
 - makes calculations and provides actual results using PyBEST quantum chemistry library,
 - knows available computational tools and libraries best,
 - returns code and its output analysis.

**VizCreator**
 - creates plots (they are send directly to user, no need to handle them yourself),
 - requires numerical input data to plot with description of variables and purpose of figure,
 - ask only after you obtain data from other experts.

Your job is:
1. Create step-by-step plan for research project. If you are not sure if your idea for a task is feasible, you can ask any expert if he can perform it and if he has any suggestions.
2. Assign tasks for experts. Review results of each task before you do next step. If the results are not satisfying, you modify task and assign a task once again with modified requirements.
3. Write all the assumptions and reasoning that you make as you go with research project.
4a. If calculations were performed successfuly: Prepare a final answer that has a structure of scientific publication that contains abstract, introduction, theory, computational details, results, conclusions, and references.
4b. If calculations were not performed successfully: Prepare a final answer that contains project description, research plan, theoretical background, necessary code snippets and list of further requirements to progress.

Never answer with question.
All the planning steps, intermediate results, and thinking should be formatted in HTML.
The final response must be formatted as markdown with latex equations enclosed by $$.
Here is a research project you are assigned with:
"""


@tool
def LiteratureReview(question: str) -> dict:
    """Search for information."""
    logger.info("Asking Literature Sage: %s" % question)
    return literature_reviewer.invoke({"messages": [HumanMessage(question)]})["messages"][
        -1
    ].content


@tool
def ScientificComputing(question: str) -> list[str]:
    """Write and execute code."""
    logger.info("Asking Code Mage: %s" % question)
    return calculation_mage.invoke({"messages": [HumanMessage(question)]})["messages"][
        -1
    ].content


@tool
def VizCreator(question: str) -> list[str]:
    """Creates interactive pictures and saves them to registry."""
    logger.info("Asking VizCreator: %s" % question)
    viz_creator.invoke({"messages": [HumanMessage(question)]})["messages"][-1].content


model_principal_investigator = ChatOpenAI(
    model=os.environ["MODEL_PRINCIPAL_INVESTIGATOR"]
)
principal_investigator = create_agent(
    model_principal_investigator,
    tools=[
        LiteratureReview,
        ScientificComputing,
        VizCreator,
    ],
    system_prompt=SystemMessage(content=[{"type": "text", "text": PI_PROMPT}]),
    debug=True,
)
