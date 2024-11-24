from dotenv import load_dotenv
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain

load_dotenv()

chat_history = []

prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the questions based only on the following context {context} and chat history",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

llm = OpenAI()

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

retriever = vector_store.as_retriever()

history_aware_retriever = create_history_aware_retriever(llm, retriever, prompt_template)

question_answer_chain = create_stuff_documents_chain(llm, prompt_template)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def ask_question(user_query):
    chain_response = rag_chain.invoke({"chat_history": chat_history, "input": user_query})

    chat_history.extend([HumanMessage(content=user_query), chain_response["answer"]])

    return chain_response["answer"]


while True:
    query = input("Enter your question here.")

    if query.lower() == "exit":
        break

    response = ask_question(query)

    print(response)

