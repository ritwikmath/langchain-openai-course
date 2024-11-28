from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = OpenAI()

prompt = ChatPromptTemplate.from_template("How to say {input} in language {language}?")

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

response = chain.invoke({
    "input": "How many states are there in USA?",
    "language": "German"
})

print(response)
