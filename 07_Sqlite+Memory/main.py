from config import model
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage 
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

class chatbot_state(TypedDict):
    message: Annotated[list[BaseMessage] , add_messages]

def chat_node(state:chatbot_state):
    message = state["message"]

    response = model.invoke(message)
    return {"message":response}

graph = StateGraph(chatbot_state)

graph.add_node("chat_node" , chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

con = sqlite3.connect(database="chatbot.db" , check_same_thread=False)
checkpoint = SqliteSaver(conn=con)

workflow = graph.compile(checkpointer=checkpoint)

config = {"configurable":{"thread_id":"2"}}
res = workflow.invoke({"message":"what was the third last user message?"} ,config=config)
for message in res["message"]:
    print(message.content)

