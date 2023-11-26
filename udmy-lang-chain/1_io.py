from langchain.llms import OpenAI
from dotenv import load_dotenv, find_dotenv

dotenv_config = find_dotenv()
_ = load_dotenv(dotenv_config)

llm = OpenAI()
result = llm.generate(['Here is a fact about Mars', 'Here is a fact about Jupiter'])

print(result.schema())
print()
print(result.llm_output)

print(result.generations)

#########################################################################################################

from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI()
chat([
    SystemMessage(content='You are a lazy teenager who just wants to party'),
    HumanMessage('Here is a fact about Mars'), HumanMessage('Here is a fact about Jupiter')])

import langchain
from langchain.cache import InMemoryCache

langchain.llm_cache = InMemoryCache()
llm.predict('Here is a fact about Mars')
llm.predict('Here is a fact about Mars')  # serves cached result

#########################################################################################################

from langchain import PromptTemplate

no_input_prompt = PromptTemplate(input_variables=[], template='Tell me a fact')
no_input_prompt.format()

single_input_prompt = PromptTemplate(input_variables=["topic"], template='Tell me a fact about  {topic}')
single_input_prompt.format(topic='Mars')

#########################################################################################################

from langchain.prompts import ChatPromptTemplate, PromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, \
    HumanMessagePromptTemplate
from langchain.schema import AIMessage, HumanMessage, SystemMessage

system_template = "You are an AI recipie assitant that specializes in {dietary_preference} dishes that can be prepared in {cooking_time} minutes. You are helping a user find a recipe for {recipe_name}."
system_prompt = SystemMessagePromptTemplate.from_template(system_template)
# system_prompt.format(dietary_preference='vegetarian', cooking_time=30, recipe_name='pasta')

human_template = "{recipe_request}"
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate(system_prompt=system_prompt, human_prompt=human_prompt)
prompt = chat_prompt.format_prompt(
    cooking_time='50 min', recipe_request='Quick snack', dietary_preference='lactose-free'
).to_messages()
result = chat(prompt)
print(result.content)

#########################################################################################################

# AI BOT LEGAL -> Simple Terms

system_template = "You are a helpful legal assistant that translates complex legal terms into plain and understandable language. You can describe content so that 10th grader can understand and act upon it."
system_prompt = SystemMessagePromptTemplate.from_template(system_template)
# FEW SHOT
# INPUT HUMAN
# OUTPUT AI
legal_text = "The provisions herein shall be severable, and if any provision or portion thereof is deemed invalid, illegal, or unenforceable by a court of competent jurisdiction, the remaining provisions or portions thereof shall remain in full force and effect to the maximum extent permitted by law."
example_input_one = HumanMessagePromptTemplate.from_template(legal_text)

plain_text = "The rules in this agreement can be separated."
example_output_one = AIMessagePromptTemplate.from_template(plain_text)

human_template = "{legal_text}"
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_prompt, example_input_one, example_output_one, human_prompt])
chat_prompt.input_variables  # defines the required vars!
example_legal_text = "...."

request = chat_prompt.format_prompt(legal_text=legal_text).to_messages()
response = chat(request)

#########################################################################################################

# Parsing output - steps - Import Parser > Format Instructions > Parse the response
from langchain.output_parsers import CommaSeparatedListOutputParser

output_parser = CommaSeparatedListOutputParser()
output_parser.format_instructions()  # shows the instructions - e.g. 'Your response should ...'
output_parser.parse(response)  # parses the output

human_template = "{request}\n{format}"
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
request = chat_prompt.format_prompt(request='Show me the money',
                                    format=output_parser.get_format_instructions()).to_messages()
result = chat(request)
output_parser.parse(result)

#########################################################################################################

from langchain.output_parsers import DatetimeOutputParser

output_parser = DatetimeOutputParser()
output_parser.get_format_instructions()  # 'Write a date in the following pattern ...'

template_text = "{request}\n{format}"
human_prompt = HumanMessagePromptTemplate.from_template(template_text)
chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
model_request = chat_prompt.format_prompt(request='What date was the 13th amendment ratified?',
                                          format_instructions=output_parser.get_format_instructions()).to_messages()
result = chat(model_request, temperature=0)
output_parser.parse(result)

from langchain.output_parsers import OutputFixingParser

misformatted = result.content
new_parser = OutputFixingParser.from_llm(parser=output_parser,
                                         llm=chat)  # this parser re-sends the request with more details

## this would produce better results
system_prompt = SystemMessagePromptTemplate.from_template('You only reply in datetime format')
chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
model_request = chat_prompt.format_prompt(request='What date was the 13th amendment ratified?',
                                          format_instructions=output_parser.get_format_instructions()).to_messages()
result = chat(model_request, temperature=0)
output_parser.parse(result)

#########################################################################################################

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class Scientist(BaseModel):
    name: str = Field(description='The name of a scientist')
    discoveries: list[str] = Field(description='A list of discoveries made by the scientist')


parser = PydanticOutputParser(pydantic_object=Scientist)
parser.get_format_instructions()  # Specifies how to format the response as JSON that can be parsed into a Scientist class instance

human_prompt = HumanMessagePromptTemplate.from_template("{request}\n{format}")
chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
request = chat_prompt.format_prompt(request='Tell me about a famouse scientist?',
                                    format=parser.get_format_instructions()).to_messages()
result = chat(request, temperature=0)
scientist = parser.parse(result)  # returns a Scientist instance

#########################################################################################################

from langchain import PromptTemplate
prompt = PromptTemplate(input_variables=['planet'], template='Tell me a fact about a planet {planet}')
prompt.save('planet_fact_prompt.json')

from langchain.prompts import load_prompt
loaded_prompt = load_prompt('planet_fact_prompt.json')

