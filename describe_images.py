import json, base64, os, re
from dotenv import load_dotenv
load_dotenv()
from groq import Groq

MODEL = "qwen/qwen3.8-27b"
client = Groq(api_key=os.environ["GROQ_API_KEY"])

recs = json.load(open("images_manifest.json"))
for r in recs:
    b64 = base64.b64encode(open(r["image_path"], "rb").read()).decode()
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Describe this image from study notes in 2-3 sentences. Say whether it is a diagram, code screenshot, or other. Mention key labels or concepts."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]}],
        )
        text = resp.choices[0].message.content or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        r["description"] = text
        print(r["image_path"], "->", text[:200])
    except Exception as e:
        r["description"] = None
        print(r["image_path"], "FAILED:", e)

json.dump(recs, open("images_manifest.json", "w"), indent=2)
