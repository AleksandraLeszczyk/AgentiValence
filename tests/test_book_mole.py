import logging
from app.book_mole import book_mole
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger()


def test_book_mole():
    messages = book_mole.invoke(
        {"messages": [HumanMessage("When pCCD orbitals are better tha RHF orbitals?")]}
    )
    logger.info("Book Mole answered: %s" % messages)
    answer = messages["messages"][-1]
    assert isinstance(answer, AIMessage)
    assert len(answer.content) > 1
    important_keywords = ["strong", "correlation", "degener", "multireference"]
    found_keywords = []
    for i in important_keywords:
        if i in answer.content:
            found_keywords.append(i)
    assert (
        len(found_keywords) > 1
    ), f"Found insufficient number of keywords: {found_keywords}"
