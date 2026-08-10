from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv


load_dotenv()
api_key = os.getenv("groq_api_key")

model = ChatGroq(model='llama-3.1-8b-instant' , api_key=api_key)

