from datetime import datetime

from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import DatetimeOutputParser
from langchain.output_parsers import OutputFixingParser
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

dotenv_config = find_dotenv()
_ = load_dotenv(dotenv_config)


class HistoryQuiz():

    def create_history_question(self, topic):
        '''
        This method should output a historical question about the topic that has a date as the correct answer.
        For example:

            "On what date did World War 2 end?"

        '''
        human_template = (
            "Your task is to create an easy to answer quiz question about ```{topic}``` that has a complete specific"
            " date as the correct answer.\n"
            "The question should be limited to one sentence only.")
        human_prompt = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
        request = chat_prompt.format_prompt(topic=topic).to_messages()

        chat = ChatOpenAI()
        result = chat(request)
        return result.content

    def get_AI_answer(self, question):
        '''
        This method should get the answer to the historical question from the method above.
        Note: This answer must be in datetime format! Use DateTimeOutputParser to confirm!

        September 2, 1945 --> datetime.datetime(1945, 9, 2, 0, 0)
        '''

        output_parser = DatetimeOutputParser()
        system_prompt = SystemMessagePromptTemplate.from_template('You answer quiz question with just a date')
        human_prompt = HumanMessagePromptTemplate.from_template("Answer the user's question: ```{request}```\n{format}")

        chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
        request = chat_prompt.format_prompt(request=question,
                                            format=output_parser.get_format_instructions()).to_messages()
        chat = ChatOpenAI()
        result = chat(request, temperature=0)

        try:
            response = output_parser.parse(result.content)
            return response
        except ValueError:
            new_parser = OutputFixingParser.from_llm(parser=output_parser, llm=chat)
            response = new_parser.parse(result.content)
            return response

    def get_user_answer(self, question):
        '''
        This method should grab a user answer and convert it to datetime. It should collect a Year, Month, and Day.
        You can just use input() for this.
        '''
        year = input("Please enter the year  : ")
        month = input("Please enter the month : ")
        day = input("Please enter the day   : ")
        return datetime(int(year), int(month), int(day))

    def check_user_answer(self, user_answer, ai_answer):
        '''
        Should check the user answer against the AI answer and return the difference between them
        '''

        return (user_answer - ai_answer).days


quiz_bot = HistoryQuiz()
question = quiz_bot.create_history_question(topic='Russian History')
print(question)

ai_answer = quiz_bot.get_AI_answer(question)
user_answer = quiz_bot.get_user_answer(question)
diff = quiz_bot.check_user_answer(user_answer, ai_answer)
if (diff == 0):
    print("You've got it!")
else:
    print("You are off by ", diff, " days.")
