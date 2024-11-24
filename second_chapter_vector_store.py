# Import the load_dotenv function to load environment variables from a .env file
from dotenv import load_dotenv
# Import FAISS vector store from the langchain_community.vectorstores module
from langchain_community.vectorstores import FAISS
# Import OpenAIEmbeddings from the langchain_openai module
from langchain_openai import OpenAIEmbeddings
# Import ChatPromptTemplate from the langchain_core.prompts module
from langchain_core.prompts import ChatPromptTemplate
# Import RunnablePassthrough from the langchain_core.runnables module
from langchain_core.runnables import RunnablePassthrough
# Import OpenAI model from the langchain_openai module
from langchain_openai import OpenAI
# Import StrOutputParser from the langchain_core.output_parsers module
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from the .env file
load_dotenv()

# Define a template for the prompt that includes context and question placeholders
template = """Answer the questions based only on the following context:  
{context}  

Question: {question}  
"""

# Initialize the OpenAI model
llm = OpenAI()

# Create a chat prompt template using the defined template
prompt = ChatPromptTemplate.from_template(template=template)

# Create a FAISS vector store from a list of texts and OpenAIEmbeddings
vector_store = FAISS.from_texts(["Ritwik works at Global Logic"], embedding=OpenAIEmbeddings())

# Define the question to ask
question = "Where does Ritwik work?"

# Create a retriever from the vector store
retriever = vector_store.as_retriever()

# Initialize a string output parser to handle the model's response
parser = StrOutputParser()

# Create a chain that combines the retriever, prompt, model, and parser
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)

# Invoke the chain with the question and get the response
response = chain.invoke(question)

# Print the response
print(response)

vector_store.add_texts(["Ritwik is a Software Developer"])

question = "What is his designation?"

response = chain.invoke(question)

print(response)
