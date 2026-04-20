"""Example: Instrument OpenAI with automatic LLM span capture.

Requires:
    pip install layerlens[openai]
    export LAYERLENS_STRATIX_API_KEY="your-api-key"
    export OPENAI_API_KEY="your-openai-key"
"""

import openai
from layerlens import Stratix
from layerlens.instrument import span, trace
from layerlens.instrument.adapters.providers.openai import instrument_openai

client = Stratix()
openai_client = openai.OpenAI()

# Instrument the OpenAI client — all chat.completions.create calls
# inside a @trace will generate LLM spans automatically.
instrument_openai(openai_client)


@trace(client)
def qa_agent(question: str):
    """Simple Q&A agent with a retrieval step and an LLM call."""

    # Manual span for a retrieval step
    with span("retrieve", kind="retriever") as s:
        # In a real app, this would query a vector database
        docs = ["Python is a programming language.", "It was created by Guido van Rossum."]
        s.output = docs

    # The OpenAI call is automatically instrumented — no span() needed
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": f"Answer using this context: {docs}"},
            {"role": "user", "content": question},
        ],
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    answer = qa_agent("What is Python and who created it?")
    print(f"Answer: {answer}")
