from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv, find_dotenv
from langchain.retrievers.multi_query import MultiQueryRetriever

dotenv_config = find_dotenv()
_ = load_dotenv(dotenv_config)

embedding_function = OpenAIEmbeddings()


def load_data():
    return Chroma(persist_directory='./data/us_const', embedding_function=embedding_function)


def vectorize_data():
    # LOAD "data/US_Constitution in a Document object
    loader = TextLoader("data/US_Constitution.txt")
    documents = loader.load()

    # Split the document into chunks (you choose how and what size)
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
    docs = text_splitter.split_documents(documents)

    # EMBED THE Documents (now in chunks) to a persisted ChromaDB
    db = Chroma.from_documents(docs, embedding_function, persist_directory='./data/us_const')
    db.persist()


def us_constitution_helper(question):
    '''
    Takes in a question about the US Constitution and returns the most relevant
    part of the constitution. Notice it may not directly answer the actual question!

    Follow the steps below to fill out this function:
    '''

    db = load_data()

    # Use ChatOpenAI and ContextualCompressionRetriever to return the most
    # relevant part of the documents.
    llm = ChatOpenAI(temperature=0)
    compressor = LLMChainExtractor.from_llm(llm)
    # retriever = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=llm)
    retriever = db.as_retriever()
    compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
    compressed_docs = compression_retriever.get_relevant_documents(query=question)
    return compressed_docs[0].page_content


# vectorize_data()

print(us_constitution_helper("What is the 1st Amendment?"))
