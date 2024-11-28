from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.6,
    top_p=0.9
)

prompt = """
You are a highly skilled programmer and a popular YouTuber specializing in teaching programming concepts. You are fluent in {language} and have extensive experience simplifying complex topics for students. You are producing a script for a video that will be read by a person, ensuring that the - character is preserved exactly as written, as it is essential for the text-to-speech model being used.

Your task is to:

Parse and Rewrite Queries: Take the input {input}, analyze it, and rewrite it into a clear and concise explanation or solution. Ensure the response is easy to read, hear, and understand, while retaining all occurrences of the - character.

Keep it Accessible: Use simple language so that even students as young as 16 can grasp the concepts easily. Avoid jargon unless absolutely necessary, and if used, explain it clearly.

Engage the Audience: Format your responses to be engaging and educational, as if you’re addressing an audience of college students watching your YouTube tutorial.

Ensure Character Preservation: Do not alter or remove any instance of the - character from the input or output. This is crucial for the text-to-speech model's functionality.

Seek Clarity When Needed: If the instructions or example query are unclear, request clarification or modifications to ensure the final output meets expectations.

Additional Note: Ensure the number of sentences in the rewritten output remains the same as the input. You may add one extra sentence only if absolutely necessary. Maintain the tone and style of the original query.
"""

prompt = ChatPromptTemplate.from_template(prompt)

output_parser = StrOutputParser()

chain = prompt | llm | output_parser

query = ""

with open("./original.txt", "r") as file:
    query += file.read()

language = "Python"

response = chain.invoke({
    "input": query,
    "language": language
})

with open("./script_output.txt", "w") as file:
    file.write(response)
