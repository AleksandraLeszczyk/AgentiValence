import logging
import os
from typing import Literal, Union

import seaborn as sns
import matplotlib.pyplot as plt
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


logger = logging.getLogger()

# In a real environment, this URL would point to your actual MCP server
MODEL = os.environ["MODEL_VIZ_CREATOR"]


PROMPT_VIZ_CREATOR = """
You are data analyst with expertise in vizualization and quantum chemistry.
You create interactive plots with description using tools.
Then tools don't return image directly but save them in the registry.
You don't access registry but get update if figure creation succeeded.
If there is an error message, try to fix it.

Example 
question: Plot potential energy surface x=1.6, 1.7, 1.8, e=-99, -99.1, -99
your steps:
  create_interactive_plot(
    data = \{'r': [1.6, 1.7, 1.8], 'energy': [-99, -99.1, -99]\},
    x='r',
    y='energy',
    title='Potential energy surface',
    description='Potential energy surface.'
    )
  answer with bool returned by tool.
"""

# @tool
# def create_plot(
#     data: dict[str, list[Union[str, float]]],
#     plot_type: Union[list[str], str] = "line",
#     x: str = None,
#     y: Union[str, list[str]] = None,
#     color: str = None,
#     title: str = "Figure",
#     description: str = ""
# ):
#     """
#     Generates a Seaborn plot based on the specified parameters.
#     """
#     sns.set(font_scale=2)
#     # Convert dict to DataFrame
#     df = pd.DataFrame(data)

#     # Handle plot_type if it's passed as a list (taking the first element)
#     if isinstance(plot_type, list):
#         plot_type = plot_type[0]

#     # Set the visual style
#     sns.set_theme(style="whitegrid")
#     plt.figure(figsize=(10, 6))

#     # Logic for handling multiple Y columns
#     # If Y is a list, we melt the dataframe to make it "long-form" for Seaborn
#     if isinstance(y, list) and len(y) > 1:
#         id_vars = [x] if x else []
#         if color and color not in y:
#             id_vars.append(color)

#         df = df.melt(id_vars=id_vars, value_vars=y, var_name="Variable", value_name="Value")
#         y_col = "Value"
#         hue_col = "Variable" if color is None else color
#     else:
#         y_col = y if isinstance(y, str) else (y[0] if y else None)
#         hue_col = color

#     # Plot Dispatcher
#     try:
#         if plot_type == "line":
#             ax = sns.lineplot(data=df, x=x, y=y_col, hue=hue_col)
#         elif plot_type == "bar":
#             ax = sns.barplot(data=df, x=x, y=y_col, hue=hue_col)
#         elif plot_type == "scatter":
#             ax = sns.scatterplot(data=df, x=x, y=y_col, hue=hue_col)
#         elif plot_type == "histogram":
#             ax = sns.histplot(data=df, x=x or y_col, hue=hue_col, kde=True)
#         elif plot_type == "box":
#             ax = sns.boxplot(data=df, x=x, y=y_col, hue=hue_col)
#         elif plot_type == "area":
#             ax = sns.lineplot(data=df, x=x, y=y_col, hue=hue_col)
#             plt.fill_between(df[x], df[y_col], alpha=0.3)
#         else:
#             raise ValueError(f"Unsupported plot type: {plot_type}")

#         # Formatting
#         plt.title(title, fontsize=15, pad=20)
#         plt.xlabel(x if x else "")
#         plt.ylabel(y_col if isinstance(y_col, str) else "")

#         # Display description below the plot
#         if description:
#             plt.figtext(0.5, -0.05, description, wrap=True, horizontalalignment='center', fontsize=10, style='italic')

#         plt.tight_layout()
#         fig = ax.get_figure()
#         figures = [i for i in os.listdir("artifacts") if i.startswith("fig") and i.endswith("png")]
#         fig.savefig(f"artifacts/fig{len(figures)}.png")
#         return f"Success. Saved figure as artifacts/fig{len(figures)}.png"

#     except Exception as e:
#         return f"An error occurred while plotting: {e}"


@tool
def create_interactive_plot(
    data: dict[str, list[Union[str, float]]],
    plot_type: Literal["line", "bar", "scatter", "histogram", "box", "area"] = "line",
    x: str = None,
    y: str | list[str] = None,
    color: str = None,
    title: str = "Interactive Plot",
) -> bool:
    """
    Creates an interactive Plotly figure from a pandas DataFrame.

    Args:
        data:      dictionary with keys being columns names and values being column values
        plot_type: One of 'line', 'bar', 'scatter', 'histogram', 'box', 'area'.
        x:         Column name for the x-axis (uses index if None).
        y:         Column name(s) for the y-axis (uses all numeric cols if None).
        color:     Column name used to color-code the series.
        title:     Chart title.

    Returns:
        True if success.
    """
    try:
        df = pd.DataFrame(data)

        # Fall back to numeric columns when y is not specified
        if y is None:
            y = df.select_dtypes(include="number").columns.tolist()

        # Use index as x-axis if not specified
        if x is None:
            df = df.copy()
            df["__index__"] = df.index
            x = "__index__"

        plot_fn = {
            "line": px.line,
            "bar": px.bar,
            "scatter": px.scatter,
            "histogram": px.histogram,
            "box": px.box,
            "area": px.area,
        }

        if plot_type not in plot_fn:
            raise ValueError(
                f"Unsupported plot_type '{plot_type}'. " f"Choose from: {list(plot_fn)}"
            )

        kwargs = dict(data_frame=df, x=x, title=title)

        # histogram only accepts a single column for x — skip y/color
        if plot_type == "histogram":
            kwargs["x"] = y[0] if isinstance(y, list) else y
            if color:
                kwargs["color"] = color
        else:
            kwargs["y"] = y
            if color:
                kwargs["color"] = color

        fig = plot_fn[plot_type](**kwargs)

        fig.update_layout(
            template="plotly_white",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        # Save to artifacts
        figures = [i for i in os.listdir("artifacts") if i.startswith("fig") and i.endswith("png")]
        fig.write_html(f"artifacts/fig{len(figures)}.html")
        return True
    
    except Exception:
        return False


model_viz_creator = ChatOpenAI(temperature=0, model_name=MODEL)

viz_creator = create_agent(
    model_viz_creator,
    tools=[create_interactive_plot],
    system_prompt=SystemMessage(
        content=[
            {
                "type": "text",
                "text": PROMPT_VIZ_CREATOR,
            }
        ]
    ),
)
