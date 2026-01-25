from openai import OpenAI
from doc_ai import DocAI
from vectordb import VectorStore


class OpenAIConfig:
    def __init__(self, api_key: str = "api", model: str = "gpt-5", user_id: str = "default_user", case_id: str = "default_case"):
        self.api_key = api_key
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        self.docai = DocAI(self.client)
        index_path = f"db/{user_id}_{case_id}_index.faiss"
        meta_path = f"db/{user_id}_{case_id}_meta.pkl"
        self.vstore = VectorStore(dim=3072, index_path=index_path, meta_path=meta_path)
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
        self.summary = ""

    def update_summary(self, history):
        if len(history) < 6:
            return

        convo = "\n".join([f"{m['role']}: {m['content']}" for m in history[-6:]])

        try:
            r = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Summarize the conversation for memory."},
                    {"role": "user", "content": convo}
                ],
                max_completion_tokens=500,
                temperature=1
            )

            content = r.choices[0].message.content
            
            if content:
                self.summary += "\n" + content
                print("Updated summary:", self.summary)
            else:
                print("Summary content was empty")
        except Exception as e:
            print(f"Error in update_summary: {e}")



    def get_stream_response(self, history):
        short_history = history[-6:]

        messages = list(self.system_prompt)

        if self.summary:
            messages.append({
                "role": "system",
                "content": f"Conversation memory:\n{self.summary}"
            })

        messages += short_history
        
        response_stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=1,
            stream=True
        )
        for chunk in response_stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    def add_attachment(self, uploaded_file):
        file_bytes = uploaded_file.read()
        filename = uploaded_file.name
        content_type = uploaded_file.type or ""

        content = self.docai.add_attachment(filename, content_type, file_bytes)

        embedding = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=content
        )
        vector = embedding.data[0].embedding

        self.vstore.add([vector], [f"{filename}: {content}"])

        return f"{filename} processed and stored in vector DB"


    def get_premium_stream_response(self, prompt, history):
        embedding = self.client.embeddings.create(
            model="text-embedding-3-large",
            input=prompt
        )
        query_vector = embedding.data[0].embedding

        top_contexts = self.vstore.search(query_vector, top_k=3)
        
        short_history = history[-6:]

        messages = list(self.system_prompt)
        
        if self.summary:
            messages.append({
                "role": "system",
                "content": f"Conversation memory:\n{self.summary}"
            })
        
        if top_contexts:
            context_text = "\n\n".join(top_contexts)
            messages.append({
                "role": "system",
                "content": f"Relevant document context:\n{context_text}"
            })
        messages += short_history + [{"role": "user", "content": prompt}]


        response_stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=1,
            stream=True
        )
        for chunk in response_stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content