"""Example: Instrument a LangChain chain with automatic span capture.

Requires:
    pip install layerlens[langchain] langchain-openai
    export LAYERLENS_STRATIX_API_KEY="your-api-key"
    export OPENAI_API_KEY="your-openai-key"
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from layerlens import Stratix
from layerlens.instrument.adapters.frameworks.langchain import LangChainCallbackHandler

client = Stratix()
handler = LangChainCallbackHandler(client)

# Build a simple chain
prompt = ChatPromptTemplate.from_template("Answer this question concisely: {question}")
llm = ChatOpenAI(model="gpt-4o")
chain = prompt | llm | StrOutputParser()

if __name__ == "__main__":
    # The callback handler captures the full chain execution as a trace
    result = chain.invoke(
        {"question": "What is retrieval-augmented generation?"},
        config={"callbacks": [handler]},
    )
    print(f"Answer: {result}")
