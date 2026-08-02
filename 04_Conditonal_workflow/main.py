from langchain_groq import ChatGroq
from langgraph.graph import StateGraph , START , END
from pydantic import BaseModel , Field
from dotenv import load_dotenv
from typing import TypedDict , Literal 
import os


load_dotenv()
api_key = os.getenv("groq_api_key")


model = ChatGroq(model='llama-3.1-8b-instant',
                 api_key=api_key)

class review_schema(BaseModel):
    sentiment:Literal["positive" , "negative"] = Field(description="Find the Sentiment")

class diagnose_schema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(description='The category of issue mentioned in the review')
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(description='The emotional tone expressed by the user')
    urgency: Literal["low", "medium", "high"] = Field(description='How urgent or critical the issue appears to be')

str_output = model.with_structured_output(review_schema)
str_output2 = model.with_structured_output(diagnose_schema)

class review_state(TypedDict):
    review:str
    sentiment:Literal["positive" , "negative"]
    diagnose:dict
    result:str


def find_sentiment(state:review_state):
    review = state["review"]
    prompt = f"For the following Review, Find the Sentiment of this {review}"
    output = str_output.invoke(prompt)
    return{"sentiment":output.sentiment}

def check_sentiment(state:review_state)-> Literal["positive_sentiment" , "run_diagnose"]:
    if state["sentiment"] == "positive":
        return "positive_sentiment"
    else:
        return "run_diagnose"

def positive_sentiment(state:review_state):
    # sentiment = state["sentiment"]
    prompt = f"""Write a warm thank-you message in response to this review:
    \n\n\"{state['review']}\"\n
    Also, kindly ask the user to leave feedback on our website."""
    response = model.invoke(prompt).content
    return {"result":response}

def run_diagnose(state:review_state):
    prompt = f"""Diagnose this negative review:\n\n{state['review']}\n"
    "Return issue_type, tone, and urgency."""
    response = str_output2.invoke(prompt)
    return {"diagnose":response.model_dump()}

def negative_sentiment(state:review_state):
    diagnose = state["diagnose"]
    prompt = f"""You are a support assistant.
    The user had a '{diagnose['issue_type']}' issue, sounded '{diagnose['tone']}', 
    and marked urgency as '{diagnose['urgency']}'.
    Write an empathetic, helpful resolution message.
    """
    response = model.invoke(prompt).content

    return {'result': response}


graph = StateGraph(review_state)

graph.add_node("Find_Sentiment" , find_sentiment)
graph.add_node("positive_sentiment" , positive_sentiment)
graph.add_node("run_diagnose" , run_diagnose)
graph.add_node("negative_sentiment" , negative_sentiment)

graph.add_edge(START , "Find_Sentiment")
graph.add_conditional_edges("Find_Sentiment" , check_sentiment)
# graph.add_edge("Find_Sentiment" , "positive_segment")
graph.add_edge("positive_sentiment" , END)
# graph.add_edge("Find_Sentiment" , "run_diagnose")
graph.add_edge("run_diagnose" , "negative_sentiment")
graph.add_edge("negative_sentiment" , END)


workflow = graph.compile()

intial_state = {
    "review":"Today's the weather is bad due to the Rain"
}

response = workflow.invoke(intial_state)
print(response["result"])
print(response["sentiment"])
print(response["diagnose"])