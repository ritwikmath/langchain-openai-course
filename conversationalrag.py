from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader, TextLoader
import bs4
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# loader = WebBaseLoader(
#     web_path=("https://lilianweng.github.io/posts/2023-06-23-agent",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )

loader = TextLoader("./qna.txt")

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

split_documents = text_splitter.split_documents(docs)

vector_store = FAISS.from_documents(split_documents, OpenAIEmbeddings())

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

retriever = vector_store.as_retriever()

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qna_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qna_chain)

while True:
    input_query = input("Enter your question?\n")

    if input_query == "exit":
        break

    response = rag_chain.invoke({"input": input_query})
    print(response["answer"])
