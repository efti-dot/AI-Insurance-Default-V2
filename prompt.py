from openai import OpenAI

class OpenAIConfig:
    def __init__(self, api_key: str = "api", model: str = "gpt-5"):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        self.system_prompt = [{
            "role": "system",
            "content": """You are a professional AI assistant specialized in Swedish insurance systems, including private and public insurance products such as health, home, vehicle, travel, accident, life, unemployment, and pension insurance.

Your responsibility is to provide accurate, practical, and policy-aligned guidance based on Swedish insurance regulations, standard industry practices, and consumer rights. Your answers must help users understand coverage, claims, obligations, exclusions, and next steps in real-world situations.
Response principles you must strictly follow:
1) Keep responses short, focused, and actionable. Do not over-explain. If the user needs more depth, provide it only when asked.
2) Avoid repeatedly mentioning “Sweden” unless it is necessary for legal clarity or explicitly requested by the user.
3) Always respond in the same language as the user’s message.
4) Use a natural, human, and conversational tone. Avoid sounding robotic, academic, or like an AI system.
"""
        }]

    def stream_response(self, prompt: str, history: list):
        """
        Generator that yields streamed tokens from OpenAI.
        """
        response_stream = self.client.chat.completions.create(
            model=self.model,
            messages=self.system_prompt + history,
            temperature=1,
            stream=True
        )

        for chunk in response_stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content