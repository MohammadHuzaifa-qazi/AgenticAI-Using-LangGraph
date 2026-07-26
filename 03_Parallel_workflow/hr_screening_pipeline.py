from langchain_groq import ChatGroq 
from langgraph.graph import StateGraph  , START , END
from typing import Literal, TypedDict, Annotated
from pydantic import BaseModel , Field
from dotenv import load_dotenv
import os
import operator

load_dotenv()

api_key = os.getenv("groq_api_key")

model = ChatGroq(model='llama-3.1-8b-instant',
                 api_key=api_key)

#state:
#resume,
#desciption,

#technical_score,
#communication_quality,
#red_flag,
#culture_fit


class departement_structured(BaseModel):
    score:int
    c_quality:Literal["Excellent" , "Good" , "Need_improvement"]
    c_fit:Literal["Strong",  "moderate" , "weak"]

output_structured = model.with_structured_output(departement_structured)


class departement_state(TypedDict):
    resume:str
    skills:list[str]
    job_description:str
    technical_score:int
    why_hire_me:str
    communication_quality:str
    culture_fit:str
    final_recommendation:str


graph = StateGraph(departement_state)

def technical_score(state:departement_state):
    skills = state["skills"]
    jb = state["job_description"]
    prompt = f""" Based on the following {jb} job decription.
    evaluate the {skills} and grade score between 0-100."""
    output = output_structured.invoke(prompt)
    return {"technical_score":output.score}

def communication_quality(state:departement_state):
    comm_quality = state["why_hire_me"]
    prompt = f"""Analyze the following 'Why Hire Me' paragraph {comm_quality} from a job candidate:
    Evaluate the communication quality based on:
    - Clarity and structure
    - Professional tone
    - Grammar and language errors
    - Confidence and impact."""

    output = output_structured.invoke(prompt)
    return {"communication_quality":output.c_quality}

def culture_fit(state:departement_state):
    culture_fit = state["why_hire_me"]
    prompt = f"""Analyze the following 'Why Hire Me' paragraph {culture_fit} for culture fit:
    Evaluate based on:
    - Teamwork and collaboration cues
    - Values alignment (ownership, growth, integrity, etc.)
    - Attitude and work ethic
    - Adaptability and openness"""
    output = output_structured.invoke(prompt)
    return {"culture_fit" : output.c_fit}


def final_recommendation(state:departement_state):
    score = state["technical_score"]
    culture_fit = state["culture_fit"]
    communication_quality = state["communication_quality"]
    if score >= 80 and culture_fit == "Strong" :
        return {"final_recommendation":"strong Hire"}
    elif score >= 70 and communication_quality == "Excellent":
        return {"final_recommendation":"Consider"} 
    else:
        return {"final_recommendation":"Reject"}

graph.add_node("technical_score" , technical_score)
graph.add_node("communication_quality", communication_quality)
graph.add_node("culture_fit" , culture_fit)
graph.add_node("final_recommendation" , final_recommendation)

graph.add_edge(START , "technical_score")
graph.add_edge(START,"communication_quality")
graph.add_edge(START,"culture_fit")
graph.add_edge("technical_score" , "final_recommendation")
graph.add_edge("communication_quality" , "final_recommendation")
graph.add_edge("culture_fit" , "final_recommendation")
graph.add_edge("final_recommendation" , END)

workflow = graph.compile()

input = {
    "resume":"5 years experience in Python, worked at 3 companies in last 2 years...",
    "why_hire_me":"I am passionate about coding and always deliver on time...",
    "job_description":"Looking for a Senior Full Stack Developer with Python and Next.js experience...",
    "skills":["Python" , "Typescipt" , "javascript" , "Nextjs" ,"LangGraph"]
}

res = workflow.invoke(input)
print(f"The Score based on the Skills:{res["technical_score"]}")
print(f"The communication quality is :{res["communication_quality"]}")
print(f"The culture fit is :{res["culture_fit"]}")
print(f"The final recommendation is :{res["final_recommendation"]}")
# print(res)
