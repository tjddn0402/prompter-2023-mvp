from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser

from pydantic import BaseModel, Field
from typing import List


class LawyerAnswer(BaseModel):
    legal_basis: List[str] = Field(description="Related laws about client's case")
    legal_solution: str = Field(description="Solution to client's case")
    conclusion: str = Field(description="Summarize key point in 1 sentence.")


def get_legal_help_chain(llm: ChatOpenAI) -> LLMChain:
    parser = PydanticOutputParser(pydantic_object=LawyerAnswer)
    init_chat_template = """You are AI lawyer. Your role is to give legal help your client visiting korea.

Following dialog is summary of conversation where you and your client.
summary of conversation : ```
{history}
```

client's legal inquiry : '{inquiry}'

related laws : ```
{related_laws}
```

Considering context and legal basis, help your client who doesn't know about Korean law at all.
And more, write your explain in markdown format to make it easy to understand.

{format_instruction}
"""
    prompt = PromptTemplate(
        template=init_chat_template,
        input_variables=["inquiry", "related_laws", "history"],
        partial_variables={"format_instruction": parser.get_format_instructions()},
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    chain = get_legal_help_chain(ChatOpenAI())
    chain.run("can I bring marihuana to Korea?")
