from config import model
from langgraph.graph import StateGraph , START , END
from typing import TypedDict
from pydantic import BaseModel
from langgraph.checkpoint.memory import InMemorySaver

class str_schema(BaseModel):
    question:str
    city:str
    population:int

res = model.with_structured_output(str_schema)

class simple_state(TypedDict):
    question:str
    city:str
    population:str


def city(state:simple_state):
    question = state["question"]
    prompt = f"Based on the question: '{question}', please provide the name of the city."
    response = res.invoke(prompt)
    return {"city":response.city}

def population(state:simple_state):
    city_name = state["city"]
    prompt = f"Based on the city: '{city_name}', please provide the population of that city."
    response = res.invoke(prompt)
    return {"population":response.population}


graph = StateGraph(simple_state)


graph.add_node("city" , city)
graph.add_node("population" , population)


graph.add_edge(START , "city")
graph.add_edge("city" , "population")
graph.add_edge("population" , END)

checkpoint = InMemorySaver()

workflow = graph.compile(checkpointer=checkpoint)

initial_state = {
    "question":"What is the capital of Pakistan?",
}

config = {
    "configurable":{
        "thread_id": "1"
    }
}

config2 = {
    "configurable":{
        "thread_id": "2"
    }
}
output = workflow.invoke(initial_state, config=config2)
print(output)


# workflow.get_state(config)
# print(workflow.get_state(config2))
print(list(workflow.get_state_history(config2)))