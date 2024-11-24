from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

template = """Answer the questions based only on the following context:  
{context}  

Question: {question}  
"""

docs = TextLoader("./qna.txt").load()

text_splitter = CharacterTextSplitter(
    separator=".\n",
    chunk_size=30,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False
)

split_documents = text_splitter.split_documents(docs)

vector_store = Chroma.from_documents(split_documents, embedding=OpenAIEmbeddings())

llm = OpenAI()

query = "How long I have to wait for support to respond? Only respond with waiting time"

retriever = vector_store.as_retriever()

prompt = PromptTemplate.from_template(template)

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = chain.invoke(query)

print(response)

