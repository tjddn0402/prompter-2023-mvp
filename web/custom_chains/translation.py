from pydantic import BaseModel, Field
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain import PromptTemplate
from typing import Optional, List
from langchain.chains import LLMChain


class Inquiry(BaseModel):
    inquiry: str = Field(description="고객의 문의 내용을 한국어로 번역한 내용")
    customer_type: Optional[str] = Field(
        description="관광객, 유학생, 노동자, 사업가 등 외국인 고객의 유형",
    )


def get_inquiry_translation_chain(llm) -> LLMChain:
    """외국인 고객의 문의를 한국어로 번역"""
    parser = PydanticOutputParser(pydantic_object=Inquiry)
    template = """너는 법적인 도움이 필요한 외국인의 말을 통역가야. 지금부터 외국인 고객이 하는 말을 변호사에게 한국어로 전달해줘.

외국인 고객 : "{inquiry}"

{format_instruction}
"""
    prompt = PromptTemplate(
        template=template,
        input_variables=["inquiry"],
        partial_variables={"format_instruction": parser.get_format_instructions()},
    )
    # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain


class Answer(BaseModel):
    related_laws: List[str] = Field(description="관련 법령")
    # cases: List[str] = Field(description="고객의 상황과 비슷한 사건에 대한 판례")
    advice: str = Field(description="법적 근거에 의한 법률가의 조언")


def get_answer_translation_chain(llm) -> LLMChain:
    """한국어로 답변된 법률 조언을 외국어로 번역"""
    parser = PydanticOutputParser(pydantic_object=Answer)
    template = """너는 한국어를 {tgt_lang}로 번역하는 번역가야. 법적인 도움이 필요한 외국인에게 법조인의 조언을 번역해줘.

법률가의 조언 : {legal_help}

{format_instruction}
"""
    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "tgt_lang",
            "legal_help",
        ],
        partial_variables={"format_instruction": parser.get_format_instructions()},
    )
    # llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain


if __name__ == "__main__":
    inquiry = "I wanna sue my boss because he didn't gave me my salary. What kind of laws i can refer?"
    # translator = CustomerTranslator(verbose=True)
    # answer = translator(
    #     inquiry
    # )
    # print(answer)
    inquiry_translation_chain = get_inquiry_translation_chain()
    translated_inquiry = inquiry_translation_chain.run(inquiry=inquiry)
    print(translated_inquiry)

    answer = "당신은 변호사를 선임할 수 있습니다."
    inquiry_translation_chain = get_answer_translation_chain()
    translated_answer = inquiry_translation_chain.run(
        src_lang="english", legal_help=answer
    )
    print(translated_answer)
