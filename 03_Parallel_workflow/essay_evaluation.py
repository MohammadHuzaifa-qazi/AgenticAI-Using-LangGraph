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

class Output_Schema(BaseModel):
    text_feedback: str = Field(description="Detailed feedback on the essay")
    score: int = Field(ge=1, le=10, description="Score on a scale of 1-10")

structured_model = model.with_structured_output(Output_Schema)

essay = """
Cricket is more than just a sport; it is a cultural phenomenon, 
a strategic battle of wits and skill, and a unifying force across nations. Often called the “gentleman’s game,” it combines elements of patience, athleticism, tactics, and raw emotion. Played with a bat and ball on a green field, 
cricket has evolved from an English village pastime into one of the most followed sports in the world, particularly in South Asia, Australia, England, the Caribbean, and Southern Africa.Origins and Historical Evolution
The roots of cricket can be traced back to 16th-century England. Early references to the game appear in records from the reign of King Edward I, though the modern form began taking shape in the late 1700s. 
The first known cricket club, Hambledon Club in Hampshire, played a pivotal role in formalizing the sport. By 1787, the Marylebone Cricket Club (MCC) was established in London, and its rules became the standard for the game.
"""
prompt = f"Evaluate the following {essay} and provide detailed feedback along with a score on a scale of 1-10: '"



class Essay_state(TypedDict):
    essay:str
    clarity_feedback:str
    depth_of_analysis:str
    language_feedback:str
    Indivdual_score:Annotated[list[int] ,operator.add]
    avg_score:float
    summary_feedback:str


graph = StateGraph(Essay_state)

def clarity_feedback(state:Essay_state):
    essay = state["essay"]
    prompt = f"Evaluate the clarity of thought of the following{essay} and provide feedback along with a score on a scale of 1-10:"
    output = structured_model.invoke(prompt)
    return {"clarity_feedback":output.text_feedback , "Indivdual_score":[output.score]}


def analysis_feedback(state:Essay_state):
    essay = state["essay"]
    prompt = f"Evaluate the depth of analysis of the following{essay} and provide feedback along with a score on a scale of 1-10:"
    output = structured_model.invoke(prompt)
    return {"depth_of_analysis":output.text_feedback , "Indivdual_score":[output.score]}


def language_feedback(state:Essay_state):
    essay = state["essay"]
    prompt = f"Evaluate the language use of the following{essay} and provide feedback along with a score on a scale of 1-10:"
    output = structured_model.invoke(prompt)
    return {"language_feedback":output.text_feedback , "Indivdual_score":[output.score]}


def final_feedback(state:Essay_state):
    clarity = state["clarity_feedback"]
    depth_of_analysis = state["depth_of_analysis"]
    language = state["language_feedback"]
    prompt = f"Based on the following feedback, create a summarized final feedback:\nClarity: {clarity}\nDepth of Analysis: {depth_of_analysis}\nLanguage: {language}"
    output = model.invoke(prompt).content
    avg_score = sum(state["Indivdual_score"]) / len(state["Indivdual_score"])
    return {"summary_feedback": output, "avg_score": avg_score}

graph.add_node("clarity_of_thought" ,  clarity_feedback)
graph.add_node("depth_of_analysis" ,  analysis_feedback)
graph.add_node("language_of_thought" ,  language_feedback)
graph.add_node("final" ,  final_feedback)



graph.add_edge(START , "clarity_of_thought")
graph.add_edge(START , "depth_of_analysis")
graph.add_edge(START , "language_of_thought")
graph.add_edge("clarity_of_thought" , "final")
graph.add_edge("depth_of_analysis" , "final")
graph.add_edge("language_of_thought" , "final")

workflow = graph.compile()

essay2 = """
Cricket is a game. It is played with a bat and ball. Many people like cricket. It is fun to watch and play. 
I think cricket is good.Cricket started long time ago in England. Now it is popular in India and other places. 
There are different typees like Test match, ODI and T20. T20 is fest and short. That is why many people like it.
In cricket, one team bats and other team bowls. Plyers hit the ball and run. Sometimes they get out. 
Good players scor many runs. Sachn Tendulkar was a great player. Virat Kohli is also good now.
Cricket has big matches like World Cup. When India wins, people get very happy. It makes everyone excited. But sometimes matches are boring when it rains or players play slow.
I thik cricket is nice because it brings people toether. Kids play in streets. It teaches teaork I guess. But it takes too much time sometimes. Money in cricket is also a lot now.
In the and, cricket okay game. It has good and bad points. Many people enjoy it so it is popular. I like watching cricket matches with friends. That is all about cricket.
"""

initial_state = {
    "essay":essay2,
}
result = workflow.invoke(initial_state)
print(f"Essay: {result['essay']}")
print(f"Clarity Feedback: {result['clarity_feedback']}")
print(f"Depth of Analysis Feedback: {result['depth_of_analysis']}")
print(f"Language Feedback: {result['language_feedback']}")
print(f"Summary Feedback: {result['summary_feedback']}")
print(f"Individual Scores: {result['Indivdual_score']}")
print(f"Average Score: {result['avg_score']}")
