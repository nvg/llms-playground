from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate

load_dotenv(find_dotenv())

human_prompt = HumanMessagePromptTemplate.from_template('Make a funny joke about {topic}')
chat_prompt_template = ChatPromptTemplate.from_messages([human_prompt])
chat = ChatOpenAI()

# Singular LLM chain
from langchain.chains import LLMChain

chain = LLMChain(llm=chat, prompt=chat_prompt_template)
result = chain.run(topic="Computers")
print(result)

# LLM Chain > ... > LLM Chain
from langchain.chains import SimpleSequentialChain

# Topic for a blog post > Outline > Create Blog > Create Text
llm = ChatOpenAI()

template1 = 'Give me a simple bullet point outline for a blog post on ```{topic}```'
prompt1 = ChatPromptTemplate.from_template(template1)
chain1 = LLMChain(llm=llm, prompt=prompt1)

template2 = 'Write a blog post using this outline ```{outline}```'
prompt2 = ChatPromptTemplate.from_template(template2)
chain2 = LLMChain(llm=llm, prompt=prompt2)

chain = SimpleSequentialChain(chains=[chain1, chain2],
                              verbose=True)  # Only supports single input / output -> so sub patterns don't matter
result = chain.run('Tasty and easy to cook chicken breast recipie')
print(result)

from langchain.chains import SequentialChain

# To have access to the keys / data
seq_chain = SequentialChain(chains=[chain1, chain2], input_variables=['topic'], output_variables=['outline'])
result = seq_chain('My recipe...')  # Returns a dictionary of responses

##################################################################################################

beginner_template = ('You are a physics teacher in a high-school who excels at explaining '
                     'complex terms in simple words who never assumes any prior knowledge. '
                     'Please explain: ```{question}```')

expert_template = ('You are a physics professor who explains physics topics to advanced audience members.'
                   'You can assume anyone you answer to has a PhD in Physics. '
                   'Please explain: ```{question}```')

# Route Prompt - [] Name, Description, Template

prompt_infos = [
    {'name': 'beginner', 'description': 'Answer basic physics questions', 'template': beginner_template},
    {'name': 'expert', 'description': 'Answer expert physics questions', 'template': expert_template},
]

dest_chains = {}
for p_info in prompt_infos:
    name = p_info['name']
    prompt_template = p_info['template']
    prompt = ChatPromptTemplate.from_template(template=prompt_template)
    chain = LLMChain(llm=llm, prompt=prompt)
    dest_chains[name] = chain

default_prompt = ChatPromptTemplate.from_template('{input}')
default_chain = LLMChain(llm=llm, prompt=default_prompt)

from langchain.chains.router.multi_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
destinations_str = "\n".join(destinations)

from langchain.prompts import PromptTemplate
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser

router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
router_prompt = PromptTemplate(template=router_template, input_variables=['input'], output_parser=RouterOutputParser())

from langchain.chains.router import MultiPromptChain

router_chain = LLMRouterChain.from_llm(llm, router_prompt)
chain = MultiPromptChain(router_chain=router_chain, dest_chains=dest_chains, default_chain=default_chain, verbose=True)

##################################################################################################

from langchain.chains import TransformChain


def transformer(inputs: dict) -> dict:
    updated_text = ''  # transform somehow
    return {'output': updated_text}


transform_chain = TransformChain(intput_variables=['text'], output_variables=['output'], transform=transformer)
# ... regular templating > llm

##################################################################################################

llm = ChatOpenAI(model='gpt-3.5-turbo')


class Scientist():
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name

json_schema = {'title':'Scientist', 'description': 'Info about a famous Scientist',
               'type':'object', 'properties': {'first_name': ''}}  # ...

template = 'Name a famous {country} scientist'

from langchain.chains.openai_functions import create_structured_output_chain
chat_prompt = ChatPromptTemplate.from_template(template)
chain = create_structured_output_chain(json_schema, llm, chat_prompt, verbose=True)
result = chain.run(country='American')

