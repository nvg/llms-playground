from dotenv import load_dotenv, find_dotenv
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

load_dotenv(find_dotenv())

llm = ChatOpenAI()
memory = ConversationBufferMemory()

conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
result = conversation.predict(input='Hola!')
print(result)

result = conversation.predict(input='Como te llamas?')
print(result)

result = conversation.predict(input='Puedes decirme un facto divertido sobre de algo?')
print(result)

print(memory.buffer)  # holds the history
print(memory.load_memory_variables({}))  # provides the history as a dict

# To save the history:
import pickle

pickled_string = pickle.dumps(conversation.memory)
with open('data/convo_memory.pkl', 'wb') as f:
    f.write(pickled_string)

# To load back:
new_memory_loaded = open('data/convo_memory.pkl', 'rb').read()

llm = ChatOpenAI()
reload_conversation = ConversationChain(llm=llm, memory=pickle.loads(new_memory_loaded))
print(reload_conversation.memory.buffer)

#######################################################################################################################

from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI()
memory = ConversationBufferMemory(k=1)  # # of conversations
conversation = ConversationChain(llm=llm, memory=memory)
conversation.predict(input='Conversation 1')
conversation.predict(input='Conversation 2')

print(memory.buffer)  # holds all conversations
print(memory.load_memory_variables({}))  # only loads the last K - 1 in this case

#######################################################################################################################

from langchain.memory import ConversationSummaryBufferMemory
llm = ChatOpenAI()
memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
conversation = ConversationChain(llm=llm, memory=memory)
conversation.predict(input='Conversation 1')
conversation.predict(input='Conversation 2')

print(memory.load_memory_variables({}))  # holds a summary of the conversation in the 'history' field


