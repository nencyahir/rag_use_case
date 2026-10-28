"""LLM calls: Groq for text, Gemini for page images. Both retry on rate limits."""
import os
import time

from .. import config


def _retry(fn, tries=5):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:                       # rate limit / transient network
            if i == tries - 1:
                raise
            time.sleep(2 ** i + 1)


def chat(system: str, user: str, temperature: float = 0.0) -> str:
    from groq import Groq
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set (put it in rag_backend/.env)")
    client = Groq(api_key=key)

    def call():
        r = client.chat.completions.create(
            model=config.GROQ_MODEL, temperature=temperature,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return (r.choices[0].message.content or "").strip()
    return _retry(call)


def describe_image(png: bytes, prompt: str) -> str:
    from google import genai
    from google.genai import types
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (put it in rag_backend/.env)")
    client = genai.Client(api_key=key)

    def call():
        r = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=[types.Part.from_bytes(data=png, mime_type="image/png"), prompt])
        return (r.text or "").strip()
    return _retry(call)
