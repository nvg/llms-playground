from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WikipediaLoader
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import SequentialChain, LLMChain

dotenv_config = find_dotenv()
_ = load_dotenv(dotenv_config)

email_text = open('data/email.txt').read()


def translate_and_summarize(email):
    """
    Translates an email written in a detected language to English and generates a summary.

    Args:
        email (str): The email to be processed and translated.

    Returns:
        dict: A dictionary containing the following keys:
            - 'language': The language the email was written in.
            - 'translated_email': The translated version of the email in English.
            - 'summary': A short summary of the translated email.

    Raises:
        Exception: If any error occurs during the LLM chain execution.

    Example:
        email = "Hola, ¿cómo estás? Espero que todo vaya bien."
        result = translate_and_summarize(email)
        print(result)
        # Output:
        # {
        #     'language': 'Spanish',
        #     'translated_email': 'Hello, how are you? I hope everything is going well.',
        #     'summary': 'A friendly greeting and a wish for well-being.'
        # }
    """
    llm = ChatOpenAI()

    template1 = "Return the language this email is written in:\n ```{email_text}```\n ONLY return the language it was written in."
    prompt1 = ChatPromptTemplate.from_template(template1)
    chain_1 = LLMChain(llm=llm,
                       prompt=prompt1,
                       output_key="language")

    template2 = "Translate the following text from {language} into English:\n ```{email_text}```"
    prompt2 = ChatPromptTemplate.from_template(template2)
    chain_2 = LLMChain(llm=llm,
                       prompt=prompt2,
                       output_key="translated_email")

    template3 = "Create a summary of the following text:\n ```{translated_email}```"
    prompt3 = ChatPromptTemplate.from_template(template3)
    chain_3 = LLMChain(llm=llm,
                       prompt=prompt3,
                       output_key="summary")

    seq_chain = SequentialChain(chains=[chain_1, chain_2, chain_3],
                                input_variables=['email_text'],
                                output_variables=['language', 'translated_email', 'summary'],
                                verbose=True)

    return seq_chain(email)


result = translate_and_summarize(email_text)
print(result)
