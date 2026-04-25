import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage 
from langchain_core.documents import Document

from dotenv import load_dotenv


load_dotenv(override=True)

PROMPT_LITERATURE_REVIEW = """
You are a knowledgeable scientist specialized in quantum chemistry with access to the datastore.
You answer questions related to quantum chemistry based on the sources using confident tone.
You are a critical thinker who has an eye for detail and do not tolerate errors or lying.
If you are not sure, don't make things up.
Write clearly if there is aknowledge gap.
Answer by giving a comprehensive summary based on given context.
You answer must be formatted as HTML. You can include relevant citations from context.
"""

MODEL = os.environ["MODEL_KNOWLEDGE_SUMMARY"]
DB_NAME = os.environ["DB_NAME"]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectordb.as_retriever()
model_literature_review = ChatOpenAI(temperature=0, model_name=MODEL)

@tool
def search(query: str) -> str:
    """Search for information."""
    return " ".join([f"New item: {i.metadata['source']} {i.page_content}" for i in retriever.invoke(query, k=10)])

literature_reviewer = create_agent(
    model_literature_review,
    tools=[search],
    system_prompt=SystemMessage(
        content=[
            {
                "type": "text",
                "text": PROMPT_LITERATURE_REVIEW,
            }
        ]
    ),
)




