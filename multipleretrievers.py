from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


def get_retriever(file_path: str):
    texts = TextLoader(file_path).load()

    text_splitter = CharacterTextSplitter(
        separator=".\n",
        chunk_size=100,
        chunk_overlap=0
    )

    split_documents = text_splitter.split_documents(texts)

    vector_database = Chroma.from_documents(split_documents, embedding=OpenAIEmbeddings())

    retriever = vector_database.as_retriever()

    return retriever


shoes_retriever = get_retriever("./shoes.txt")

cloths_retriever = get_retriever("./cloths.txt")

retrievers = {
    "SHOES": shoes_retriever,
    "CLOTHS": cloths_retriever
}


class Search(BaseModel):
    """Search over a dataset of Question and Answer list about Shoes and Cloths"""

    query: str = Field(
        ...,
        description="Similarity search query applied to QnA context always with question mark"
    )
    item: str = Field(
        ...,
        description="Item to look for. Should be SHOES or CLOTHS"
    )


system = """You are an expert in finding relevant information from a pool of dataset.
Your primary job is to find answers from a pool of QnA.
If there are acronyms or words you are not familiar with, do not try to rephrase them.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{query}")
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

structured_llm = llm.with_structured_output(Search)

query_analyzer = {"query": RunnablePassthrough()} | prompt | structured_llm


def retrieval(search: Search) -> str:
    retriever = retrievers[search.item]
    final_prompt = ChatPromptTemplate.from_template(
        """
        You are to find an answer from the following context:  
        {context}
        
        If the requested item is not explicitly mentioned in the context, try to identify if there is a similar item in the list that matches its category or characteristics. Provide the closest match where applicable.  
        
        Question: {question}
        """
    )
    model = ChatOpenAI()
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | final_prompt
        | model
        | StrOutputParser()
    )

    final_response = chain.invoke(search.query)

    return final_response


response = query_analyzer.invoke("What type of woman ware you have?")

output = retrieval(response)

print(output)
