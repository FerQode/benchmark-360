"""Debug script: call Gemini directly and show raw bytes to diagnose JSON parse failure."""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.processors.llm_client_factory import LLMClientFactory


async def main():
    factory = LLMClientFactory()
    fallback = factory.get_fallback_client()
    print(f"Provider: {fallback.provider_name}, model: {fallback.text_model}")

    simple_prompt = [
        {"role": "system", "content": "You are a JSON extraction assistant. Always respond ONLY with valid JSON."},
        {"role": "user", "content": 'Return this exact JSON: {"plans": [{"name": "Test Plan", "price": 25.0}], "extraction_metadata": {"total_plans_found": 1}}'},
    ]

    response = await fallback.client.chat.completions.create(
        model=fallback.text_model,
        messages=simple_prompt,
        temperature=0.0,
        max_tokens=500,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or ""
    
    print(f"\n=== Raw response ({len(raw)} chars) ===")
    print(repr(raw[:500]))
    print(f"\n=== First 10 bytes (hex): {raw[:10].encode('utf-8').hex()} ===")
    print(f"\n=== Decoded text ===")
    print(raw[:500])

    import json
    try:
        parsed = json.loads(raw)
        print(f"\n✅ json.loads SUCCESS — {len(parsed)} top-level keys")
    except json.JSONDecodeError as e:
        print(f"\n❌ json.loads FAILED: {e}")
        # Try stripping BOM
        stripped = raw.lstrip('\ufeff').strip()
        try:
            parsed = json.loads(stripped)
            print(f"✅ After BOM strip — SUCCESS ({len(parsed)} keys)")
        except:
            print("❌ Still fails after BOM strip")
        
        # Try extracting {...}
        import re
        m = re.search(r'(\{.*\})', raw, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1))
                print(f"✅ After regex extract — SUCCESS ({len(parsed)} keys)")
            except json.JSONDecodeError as e2:
                print(f"❌ regex extract also fails: {e2}")
                print(f"Extracted chunk repr: {repr(m.group(1)[:200])}")


asyncio.run(main())
