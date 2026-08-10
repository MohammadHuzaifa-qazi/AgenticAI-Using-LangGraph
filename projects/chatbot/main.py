from config import model
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage 
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

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

checkpoint = MemorySaver()

workflow = graph.compile(checkpointer=checkpoint)

thread_id = "1"
# while True:

    # user_msg = input("Type Here:")
    # print("User:", user_msg)
    # if user_msg.strip().lower() in ["exit" , "bye"]:
    #     break
    # config = {"configurable":{"thread_id":thread_id}}
    # response = workflow.invoke({"message":[HumanMessage(content=user_msg)]}, config=config)
    # print("Response:", response["message"][-1].content)

# initial_state = {
#     "message":[HumanMessage(content="Hello!")]
# }

# result = workflow.invoke(initial_state, config=config)
# print(result["message"][-1].content)
# message_dict = result["message"][-1]
# print(message_dict)   # Ye ek Python dictionary dega, jise json.dumps() se JSON string bhi bana sakte hain

