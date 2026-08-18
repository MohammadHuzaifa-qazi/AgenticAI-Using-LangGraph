from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("groq_api_key")

model = ChatGroq(model='openai/gpt-oss-20b', api_key=api_key)
