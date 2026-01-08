import io, base64
from typing import List, Dict, Optional
from PIL import Image
from openai import OpenAI

class DocAI:
    def __init__(self, client: OpenAI):
        self.client = client
        self.kb: List[Dict[str, str]] = []

    def img64_from_bytes(self, img_bytes: bytes) -> Optional[str]:
        try:
            img = Image.open(io.BytesIO(img_bytes))
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            data = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{data}"
        except Exception as e:
            print(f"Error converting image: {e}")
            return None

    def analyze_img_from_bytes(self, img_bytes: bytes, name: str, context: str = "") -> str:
        img_url = self.img64_from_bytes(img_bytes)
        if not img_url:
            return f"Could not analyze image bytes for {name}"
        r = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Describe all details from image in {name}. {context}".strip()},
                    {"type": "image_url", "image_url": {"url": img_url}}
                ]
            }],
            max_tokens=1000
        )
        return r.choices[0].message.content
    

