from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts import PromptTemplate


def get_legal_help_chain(llm) -> LLMChain:
    init_chat_template = """너는 유능한 ai 변호사야. 한국을 방문하는 외국인이 법적인 도움을 요청해서 법적인 도움을 줘야 해.

법률 문의 : {inquiry}

관련 법 조항들 : ```
{related_laws}
```

위의 법률을 참고해서 한국 법을 잘 모르는 외국인이 도움을 받을 수 있게 잘 설명해줘.
단, 고객이 글을 알아보기 쉽게 markdown 문법으로 작성해줘.
"""
    prompt = PromptTemplate(
        template=init_chat_template,
        input_variables=["inquiry", "related_laws"],
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain
