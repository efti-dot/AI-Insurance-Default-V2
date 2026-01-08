import io, base64
from typing import List, Dict, Optional
from PIL import Image
from openai import OpenAI
from docx import Document


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
    
    def text_docx_from_bytes(self, name: str, file_bytes: bytes) -> str:
        doc = Document(io.BytesIO(file_bytes))
        content_parts: List[str] = []

        # Text
        text_content = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        if text_content:
            content_parts.append(f"TEXT CONTENT:\n{text_content}")

        # Images
        img_count = 0
        for rel in doc.part.rels.values():
            if "image" in getattr(rel, "target_ref", ""):
                try:
                    img_bytes = rel.target_part.blob
                    img_desc = self.analyze_img_from_bytes(
                        img_bytes,
                        name,
                        f"This is image {img_count + 1} from the document"
                    )
                    img_count += 1
                    content_parts.append(f"\nIMAGE {img_count} DESCRIPTION:\n{img_desc}")
                except Exception as e:
                    print(f"Error processing image: {e}")

        return "\n\n".join(content_parts) if content_parts else "No content found"


