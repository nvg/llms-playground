from langchain.document_loaders import CSVLoader

loader = CSVLoader("my_file.csv")
data = loader.load()  # python list of Document objects - page content - contains a new line separated list of key/value pairs
print(data[0].page_content)  # 'key1: value1\nkey2: value2\nkey3: value3\n'

# pip install beautifulsoup4
from langchain.document_loaders import BSHTMLLoader

loader = BSHTMLLoader("my_file.html")
data = loader.load()  # python list of Document objects - page content - contains a new line separated list of key/value pairs
print(data[0].page_content)  # Contains just the text from the page

# pip install pypdf
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("my_file.pdf")
pages = loader.load()  # this loader may throw off the format - provide extra new lines, etc

######################################################################################################

# Integrations - work with servies - not always guaranteed to work, as they rely on 3rd party service
from langchain.document_loaders import HNLoader

loader = HNLoader("https://news.ycombinator.com/item?id=36697119")
data = loader.load()
print(data[0].page_content)

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI

human_prompt = HumanMessagePromptTemplate.from_template(
    "Please provide a short summary of the following comment:\n```{comment}```")
chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
model = ChatOpenAI()
result = model(chat_prompt.format_prompt(comment=data[0].page_content).to_messages())
print(result.content)

######################################################################################################

from langchain.text_splitter import CharacterTextSplitter

with open('my_text_file.txt') as file:
    text = file.read()

text_splitter = CharacterTextSplitter(separator="\n\n", chunk_size=1000)
texts = text_splitter.create_documents([text])
texts[0]

# pip install tiktoken - splits test into tokens
text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
texts = text_splitter.split_text(text)

######################################################################################################

from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
text = 'This is a string to be embedded'
embedded_text = embeddings.embed_query(text)

from langchain.document_loaders import CSVLoader

loader = CSVLoader('my_data.csv')
data = loader.load()

# [text.page_content for text in data] # converts loaded data into an list of strings
embedded_docs = embeddings.embed_documents([text.page_content for text in data])

######################################################################################################

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader

loader = TextLoader('my_text.txt')
documents = loader.load()

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
docs = text_splitter.split_documents(documents)

embedding_function = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embedding_function, persist_directory="./my_text_db")
db.persist()
# db_new_connection = Chroma(persist_directory='./my_text_db', embedding_function=embedding_function)  # to load

new_doc = "What did text say about XYZ"  # Same as "XYZ, text"
similar_docs = db.simliarity_search(new_doc)
similar_docs[0].page_content  # provides the closest match

retriever = db.as_retriever()
results = retriever.get_relevant_documents('XYZ')  # Gives a list of documents containing XYZ

######################################################################################################

from langchain.document_loaders import WikipediaLoader

loader = WikipediaLoader(query='MKUltra')
documents = loader.load()

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=500)
docs = text_splitter.split_documents(documents)

from langchain.embeddings import OpenAIEmbeddings

embedding_function = OpenAIEmbeddings()

db = Chroma.from_documents(docs, embedding_function, persist_directory='./mk_ultra')
db.persist()

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI

question = "When was this declassified?"
llm = ChatOpenAI(temperature=0)  # temp will cause repeatable queries

# Knows how to generate multiple similar queries out of the original one
retriever_from_llm = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=llm)

# Logging - just to know what happens behind the scenes
import logging

logging.basicConfig()
logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

# This will not answer the query - just returns N docs that are most similar / relevant
unique_docs = retriever_from_llm.get_relevant_documents(query=question)
print(unique_docs[0].page_content)

######################################################################################################

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

llm = ChatOpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                       base_retriever=db.as_retriever())
docs = db.simliarity_search('What is XYZ?')
docs[0]  # pull the chunk

compressed_docs = compression_retriever.get_relevant_documents('What is XYZ?')
compressed_docs[0].page_content  # pulls in summarized chunk - nice and concise!

