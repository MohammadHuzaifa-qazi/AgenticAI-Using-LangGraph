from langchain_groq import ChatGroq 
from langgraph.graph import StateGraph  , START , END
from typing import TypedDict, Annotated
from pydantic import BaseModel , Field
from dotenv import load_dotenv
import os
import operator

load_dotenv()

api_key = os.getenv("groq_api_key")

model = ChatGroq(model='llama-3.1-8b-instant',
                 api_key=api_key
                 )

class hr_schema(BaseModel):
    skills: list[str] = []
    experience: float = 0.0

str_model = model.with_structured_output(hr_schema)


class hr_state(TypedDict):
    resume:str
    skills:Annotated[list[str] , operator.add]
    experience:float
    role_suggest:str

graph = StateGraph(hr_state)

def skill_analysis(state:hr_state):
    skills = state["skills"]
    resume = state["resume"]
    output = str_model.invoke(f"Analyze the following resume {resume} and skills {skills} and provide a summary of the skills:")
    return {"skills":output.skills}

def experience_analysis(state:hr_state):
    experience = state["experience"]
    resume = state["resume"]

    output = str_model.invoke(f"Analyze the following resume {resume} and experience {experience} and provide a summary of the experience:")
    return {"experience":output.experience}

def role_suggestion(state:hr_state):
    skills = state["skills"]
    experience = state["experience"]
    role_suggest = model.invoke(f"Based on the following skills {skills} and experience {experience}, suggest a suitable role:").content
    return {"role_suggest":role_suggest}

graph.add_node("skill_analysis" , skill_analysis)
graph.add_node("experience_analysis" , experience_analysis)
graph.add_node("role_suggestion" , role_suggestion)

graph.add_edge(START , "skill_analysis")
graph.add_edge(START , "experience_analysis")
graph.add_edge("skill_analysis" , "role_suggestion")
graph.add_edge("experience_analysis" , "role_suggestion")
graph.add_edge("role_suggestion" , END)


# resume = """
# Skills (Tech Stack):
# React.js, Node.js, Express.js, MongoDB, JavaScript, TypeScript, Tailwind CSS, REST APIs, Git, Docker, AWS, Next.js, Redux

# Experience:
# 4.5 years of experience in Full Stack Web Development
# Currently working as Full Stack Developer at TechNova Solutions (2.5 years)
# Previously worked as Junior Developer at CodeSphere (2 years)

# """
resume2 = """
Skills (Tech Stack):
Python, Django, Flask, PostgreSQL, React.js, Pandas, NumPy, Machine Learning (Scikit-learn), Docker, AWS, Git, REST APIs
Experience:
Total 3+ years of experience
"""

intial_state = {
    "resume": resume2,
    "skills": [],
    "experience": 0
}

workflow = graph.compile()

response = workflow.invoke(intial_state)
print(response["skills"])
print(response["experience"])
print(response["role_suggest"])
