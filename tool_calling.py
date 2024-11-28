from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

load_dotenv()

llm = ChatOpenAI()


@tool
def full_name(f_name: str, l_name: str) -> str:
    """Concat f_name and l_name.

    Args:
        f_name: first string
        l_name: last string
    """
    return f_name + " " + l_name


functions = {
    "full_name": full_name
}


tools = [full_name]

llm_with_tools = llm.bind_tools(tools)

response = llm_with_tools.invoke("My first name is Ritwik and last name is Math. What is my full name?")

tool_calling_args = response.tool_calls[0]

# print(tool_calling_args)

print(functions[tool_calling_args["name"]].invoke(tool_calling_args["args"]))

