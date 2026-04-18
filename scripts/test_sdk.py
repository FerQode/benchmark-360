"""Full SDK + API key test after migration."""
import asyncio
import os

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types as genai_types

print("Imports: google-genai OK")


async def test_gemini():
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    payload = 'Solo JSON valido: {"status": "ok"}'
    r = await client.aio.models.generate_content(
        model="gemini-2.5-flash",
        contents=payload,
        config=genai_types.GenerateContentConfig(
            temperature=0.0,
            max_output_tokens=60,
            response_mime_type="application/json",
        ),
    )
    print("Gemini 2.5 Flash OK:", r.text[:80])


asyncio.run(test_gemini())
print("All SDK checks passed!")
