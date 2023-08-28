'''
Simple Code for chatting with GPT-3.5
'''
import openai
import os
import keyboard

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key  = os.getenv('OPENAI_API_KEY')

def get_completion():
    prompt = input("query: ")

    if prompt == "q" :
        return prompt

    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    print("answer : ", response.choices[0].message["content"])

while True :
    chat = get_completion()
    if chat == "q" :
        print("exit")
        break