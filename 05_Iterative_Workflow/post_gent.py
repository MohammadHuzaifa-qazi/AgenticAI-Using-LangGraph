from config import model
from langgraph.graph import StateGraph , START , END
from typing import Literal, TypedDict
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage


model = model
class Post_State(TypedDict):
    topic:str
    post_text:str
    evaluator:Literal["Approved" , "Need_improved"]
    feedback:str
    iteration:int
    optimizer:str
    max_iterations:int


class evaluator_schema(BaseModel):
    feedback:str
    evaluator:Literal["Approved" , "Need_improved"] 

str_result = model.with_structured_output(evaluator_schema)

def Generate_post(state:Post_State):
    topic = state["topic"]
    prompt = [SystemMessage
              (content="""You are a skilled social media content writer specializing in Pakistani national occasions. 
              Your task is to create short, high-quality posts for X (Twitter) about Pakistan’s Independence Day (14 August)."""),
            HumanMessage(
                content=f"""Based on the following topic {topic},
                Write a single Twitter/X post for {topic}.
                Tone should be professional yet funny — mix of pride and light humour, no sarcasm or insult.
                Keep it under 180 characters.
                Do not ask any question.
                No hashtags at the end if they feel forced; use only 1-2 relevant ones if natural.
                Language: Mix of simple English and Roman Urdu is allowed, but keep it clean and shareable.
                Make it feel genuine, not corporate or cringe""")]
    res = model.invoke(prompt).content
    return {"post_text":res}



def Evaluate_post(state:Post_State):
    post_text = state["post_text"]
    prompt = [SystemMessage
              (content="""You are a strict but fair content evaluator for X (Twitter) posts about Pakistan’s Independence Day (14 August).
              Your job is to review a given post and decide if it meets the required quality standards.
                2"""),
            HumanMessage(
                content=f"""Based on the following topic {post_text},
                Evaluate the following Independence Day post according to the rules:
                My Rules:
                check it under 280 characters.
                Do not ask any question.
                No hashtags at the end if they feel forced; use only 1-2 relevant ones if natural.
                Language: Mix of simple English and Roman Urdu is allowed, but keep it clean and shareable.
                Make it feel genuine, not corporate or cringe""")
                ]
    res = str_result.invoke(prompt)
    return {"evaluator":res.evaluator , "feedback":res.feedback}

def optimizer(state:Post_State):
    feedback = state["feedback"]
    topic = state["topic"]
    post = state["post_text"]
    iteration = state["iteration"]
    prompt = [SystemMessage(content="""You are an expert content optimizer for X (Twitter) posts about Pakistan’s Independence Day (14 August).
    Your task is to take a post that has been marked as “Need Improvement” along with the evaluator’s feedback, and improve it.""") , 
    HumanMessage(content=f"""Based On the following {feedback},
    P]lease optimize the post according to the feedback.
    topic:{topic} and post{post}
    """)
    ]
    response = model.invoke(prompt).content
    iteration = iteration+1
    return {"post_text":response , "iteration":iteration}

def evaluation_decision(state:Post_State):
    if state["evaluator"] == "Approved" or state["iteration"] >= state["max_iterations"]:
        return "Approved"
    else:
        return "Need_improved"

    
graph = StateGraph(Post_State)

graph.add_node("Generate_post" , Generate_post)
graph.add_node("Evaluate_post" , Evaluate_post)
graph.add_node("optimizer" , optimizer)


graph.add_edge(START ,"Generate_post")
graph.add_edge("Generate_post" , "Evaluate_post")
graph.add_conditional_edges("Evaluate_post" , evaluation_decision , {"Approved":END , "Need_improved":"optimizer"})

graph.add_edge("optimizer","Evaluate_post")


workflow = graph.compile()

intial_state = {
    "topic":"How we Celebarate the Independence day in Pakistan.",
    "iteration":1,
    "max_iterations":2
}

output = workflow.invoke(intial_state)
print(f"The Topic is: {output["topic"]}")
print(f"The Post Text: {output["post_text"]}")
print(f"The Feedback is: {output["feedback"]}")
print(f"The Evaluation is: {output["evaluator"]}")
print(f"The loop iteration:{output["iteration"]}")