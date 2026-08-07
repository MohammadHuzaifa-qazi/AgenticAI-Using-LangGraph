from config import model
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from pydantic import BaseModel
from langgraph.checkpoint.memory import InMemorySaver

class str_schema(BaseModel):
    question: str
    city: str
    population: int

res = model.with_structured_output(str_schema)

class simple_state(TypedDict):
    question: str
    city: str
    population: str

def city(state: simple_state):
    response = res.invoke(f"Based on the question: '{state['question']}', please provide the name of the city.")
    return {"city": response.city}

def population(state: simple_state):
    response = res.invoke(f"Based on the city: '{state['city']}', please provide the population of that city.")
    return {"population": response.population}

graph = StateGraph(simple_state)
graph.add_node("city", city)
graph.add_node("population", population)
graph.add_edge(START, "city")
graph.add_edge("city", "population")
graph.add_edge("population", END)

checkpoint = InMemorySaver()
workflow = graph.compile(checkpointer=checkpoint)

config = {"configurable": {"thread_id": "1"}}
workflow.invoke({"question": "What is the capital of Pakistan?"}, config=config)
print("state:", workflow.get_state(config).values)
print("history len:", len(list(workflow.get_state_history(config))))
