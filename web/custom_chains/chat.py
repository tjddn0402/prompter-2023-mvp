from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


def get_legal_help_chain(llm) -> LLMChain:
    init_chat_template = """너는 유능한 ai 변호사야. 한국을 방문하는 외국인이 법적인 도움을 요청해서 법적인 도움을 줘야 해.

아래 대화는 client와 지금까지 주고받은 대화내용을 요약한 것이다.
chat history with client : ```
{history}
```

client의 새로운 법률 문의 : '{inquiry}'

관련 법 조항들 : ```
{related_laws}
```

지금까지 주고받은 대화의 맥락을 고려하고, 위의 법률을 참고해서 한국 법을 잘 모르는 외국인이 도움을 받을 수 있게 잘 설명해줘.
단, 고객이 글을 알아보기 쉽게 markdown 문법으로 작성해줘.
"""
    prompt = PromptTemplate(
        template=init_chat_template,
        input_variables=["inquiry", "related_laws", "history"],
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain
