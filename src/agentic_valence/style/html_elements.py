import os

import html
import json
from langchain.messages import AIMessage, ToolMessage


def get_css_style_header():
    return """<style>
        .event-container { 
            word-wrap: break-word; 
            overflow-wrap: break-word; 
            white-space: normal;
            font-family: sans-serif;
            margin-bottom: 15px;
            display: none;
            overflow: hidden;
        }
        .code-block {
            white-space: pre-wrap; 
            word-break: break-all;
            background: rgba(0,0,0,0.05);
            padding: 8px;
            border-radius: 4px;
            font-size: 0.9em;
        }
        .active, .collapsible:hover {
            background-color: #ccc;
            }
        @media (prefers-color-scheme: dark) {
            .code-block { background: rgba(255,255,255,0.1); color: #e5e7eb; }
            .tool-call { border-left-color: #60a5fa !important; }
            .tool-response { background: #064e3b !important; border-color: #059669 !important; color: #ecfdf5; }
        }
    </style>
    """

def format_event_panel(new_event: AIMessage | ToolMessage, old_event_panel: str) -> str:

    # Get clean events
    new_event_html = parse_event_to_html(new_event)
    if not new_event_html.replace("<br>", ""):
        return old_event_panel
    out = f"""<br>{new_event_html}<br> {old_event_panel}"""
    return out


def parse_event_to_html(event: AIMessage | ToolMessage) -> str:
    """
    Parses a LangChain event to HTML with dark mode support and overflow protection.
    """
    content = getattr(event, "content", "")
    msg_type = type(event).__name__

    # CSS for responsiveness and theme adaptation
    html_output = ""

    if msg_type == "AIMessage":
        tool_calls = getattr(event, "tool_calls", [])
        if isinstance(content, str):
            if not content.startswith("#"):
                html_output += (
                    f'<div class="event-container tool-call" style="border-left: 4px solid #3b82f6; padding-left: 12px;">'
                    f"<b>🛠️ Action: thinking</b><br>"
                    f'{content}'
                    f"</div>"
                )
        if tool_calls:
            for tc in tool_calls:
                name = html.escape(tc.get("name", "unknown"))
                args = html.escape(json.dumps(tc.get("args", {}), indent=2))

                html_output += (
                    f'<div class="event-container tool-call" style="border-left: 4px solid #3b82f6; padding-left: 12px;">'
                    f"<b>🛠️ Action: calling <code>{name}</code></b><br>"
                    f'<b>Task:</b><pre class="code-block"><code>{args}</code></pre>'
                    f"</div>"
                )


    if msg_type == "ToolMessage":
        # Escape the name for safety, but allow 'content' to render as raw HTML
        name = html.escape(getattr(event, "name", "Expert"))

        if name == "VizCreator":
            figures = sorted([
                i for i in os.listdir("artifacts")
                if i.startswith("fig") and (i.endswith("html") or i.endswith("png"))
            ])

            html_output = (
                f'<div class="event-container tool-response" style="background: #f0fdf4; border: 1px solid #bbf7d0; padding: 12px; border-radius: 8px; color: #166534;">'
                f'<h4 style="margin: 0 0 8px 0;">🎨 Figure {len(figures)}</h4>'
                f"""<iframe src="gradio_api/file/artifacts/{figures[-1]}" width="100%" height="500px"></iframe>"""
                f"</div>"
            )

        else:
            # Other tool responses
            html_output = (
                f'<div class="event-container tool-response" style="background: #f0fdf4; border: 1px solid #bbf7d0; padding: 12px; border-radius: 8px; color: #166534;">'
                f'<h4 style="margin: 0 0 8px 0;">🧱 <code>{name}</code> results</h4>'
                f'<div style="line-height: 1.5;">{content}</div>'
                f"</div>"
            )

    return f"{html_output.strip()}" if html_output else ""