from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

dotenv_config = find_dotenv()
_ = load_dotenv(dotenv_config)

system_template = "You are an experienced travel assistant that specializes in authentic travel experience.."
system_prompt = SystemMessagePromptTemplate.from_template(system_template)

human_template = "You task is to recommend travel for {interest} that has a {budget} budget."
human_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
prompt = chat_prompt.format_prompt(
    interest='fishing', budget='$10,000'
).to_messages()

chat = ChatOpenAI()
result = chat(prompt)
print(result.content)
