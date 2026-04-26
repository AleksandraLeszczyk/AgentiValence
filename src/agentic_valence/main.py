from langchain.messages import AIMessage, HumanMessage, SystemMessage

from agentic_valence.agents.principal_investigator import principal_investigator
from agentic_valence.style.html_elements import format_event_panel


def chat_with_principal_investigator(
    history: list[dict], research_progress: str = "🔬 Research Progress\n"
):
    """
    Generator function that streams agent events to the Gradio UI.
    Yields: (updated_history, event_markdown_string)
    """
    # Convert full Gradio history to LangChain messages
    langchain_messages = []
    for msg in history:
        content = msg["content"]
        # Handle multimodal content (list with text/files)
        if isinstance(content, list):
            text = content[0]["text"]
        else:
            text = content

        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=text))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=text))
        elif msg["role"] == "system":
            langchain_messages.append(SystemMessage(content=text))

    # Initialize accumulation strings
    full_answer = ""

    # Pass the full message history instead of just the last message
    for chunk in principal_investigator.stream(
        {"messages": langchain_messages}, stream_mode="values"
    ):
        # 1. Update the Event Log (Right Panel)
        research_progress = format_event_panel(chunk["messages"][-1], old_event_panel=research_progress)

        # 2. Update the Final Answer (if available in the current chunk)
        if "messages" in chunk:
            last_msg = chunk["messages"][-1]
            if hasattr(last_msg, "content") and isinstance(last_msg.content, str):
                full_answer = last_msg.content

        # Yield current state to the UI
        yield history, research_progress

    # Final yield to lock in the result
    history.append({"role": "assistant", "content": full_answer})
    yield history, research_progress

