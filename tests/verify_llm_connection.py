import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

from src.config import BASE_URL


def test_llm_connection():
    api_key = os.getenv("OPENROUTER_API_KEY")
    llm = ChatOpenAI(
        model="stepfun/step-3.5-flash:free",
        api_key=api_key,
        base_url=BASE_URL,
        temperature=0,
    )
    resp = llm.invoke("Reply with the single word: ok")
    assert resp is not None
    assert isinstance(resp.content, str) and len(resp.content) > 0
