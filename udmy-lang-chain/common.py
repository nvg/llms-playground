from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv

dotenv_config = find_dotenv()
_ = load_dotenv(dotenv_config)
api_key = os.getenv('OPENAI_API_KEY')

client = OpenAI(
    api_key=api_key
)


def get_completion(prompt, model="gpt-3.5-turbo",
                   system_prompt='You are a skilled prompt engineer who is working on a project to improve the '
                                 'quality of the prompts.'):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content
