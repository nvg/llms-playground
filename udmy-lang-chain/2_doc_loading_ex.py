from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WikipediaLoader
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

dotenv_config = find_dotenv()
_ = load_dotenv(dotenv_config)


def answer_question_about(topic: str, question: str) -> str:
    loader = WikipediaLoader(query=topic, load_max_docs=1)
    data = loader.load()

    human_prompt = HumanMessagePromptTemplate.from_template(
        "Provided the following content:\n\n```{content}``` \n\n"
        "Please answer the following question:\n\n"
        "```{question}```")

    chat_prompt = ChatPromptTemplate.from_messages([human_prompt])
    model = ChatOpenAI()
    result = model(chat_prompt.format_prompt(content=data[0].page_content, question=question).to_messages())
    return result.content


print(answer_question_about("Habra Habr", "When was he born?"))  # Careful as loader would still provide data here
