from config import model
from langgraph.graph import StateGraph , START , END
from typing import Literal, TypedDict
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage


class category_schema(BaseModel):
    category:Literal["Payment" , "Delivery" , "Product" , "Other"]

class severity_schema(BaseModel):
    severity:Literal["High" , "Medium" , "Low" ,"Critical"]

class sentiment_schema(BaseModel):
    sentiment:Literal["frustrated" , "neutral" , "happy", "disappointed"]

class evaluator_schema(BaseModel):
    evaluator:Literal["Approved" , "Need_improved"]
    feedback:str


str_category = model.with_structured_output(category_schema)
str_severity = model.with_structured_output(severity_schema)
str_sentiment = model.with_structured_output(sentiment_schema)
str_evaluator = model.with_structured_output(evaluator_schema)

class complain_state(TypedDict):
    category:str
    complain:str
    order_id:str
    severity:str
    sentiment:str
    draft_response:str
    order_id:str  
    evaluator:str
    feedback:str 
    iteration:int
    max_iteration:int
    escalate_to_human:str
    auto_send:str


def categorize_complain(state:complain_state):
    message = [
        SystemMessage(
            content="You are a customer-support classifier. Your only job is to read a user's complaint and put it into exactly one category."
        ),
        HumanMessage(
            content=f"""
            Classify the following customer complaint into exactly ONE category:
            {state['complain']}

            Categories:
            - Payment: refund, transaction failed, double charge, payment not received, wallet, COD, online payment problems
            - Delivery: shipping, delayed delivery, order not received, wrong address, tracking, courier, package lost/damaged in transit
            - Product Issue: defective, damaged, wrong item, quality problem, missing parts, size/color mismatch
            - Other: anything that does not fit Payment, Delivery, or Product Issue (account issues, general questions, feedback, spam)

            Return only the single best-matching category.
            """
        )
    ]
    output = str_category.invoke(message)
    return {"category": output.category}

def severity_check(state:complain_state):
    complain = state["complain"]
    message = [
        SystemMessage(
            content="You are a severity-assessment agent for customer-support tickets. Assess how urgent and serious a customer's complaint is."
        ),
        HumanMessage(
            content=f"""
            Assess the severity of this customer complaint:
            Complaint: {complain}

            Classify severity into exactly ONE level:
            - Critical: Very serious (money loss, repeated failure, strong anger, legal threat, complete service failure)
            - High: Significant issue affecting the customer, should be prioritized
            - Medium: Noticeable problem, needs attention soon
            - Low: Minor issue, no major impact, can wait

            Rules:
            - Pick only one level
            - Consider both the complaint content and the emotional intensity
            - Do not solve the issue; only assess severity
            """
        )
    ]
    output = str_severity.invoke(message)
    return {"severity": output.severity}


def sentiment_check(state:complain_state):
    complain = state["complain"]
    message = [
        SystemMessage(
            content="You are a sentiment-analysis specialist for customer-support complaints. Your task is to detect the customer's overall emotional tone."
        ),
        HumanMessage(
            content=f"""
            Analyze the sentiment of this customer complaint:
            Complaint: {complain}

            Classify the sentiment into exactly ONE category:
            - happy: satisfied, appreciative
            - neutral: calm, factual, no strong emotion
            - disappointed: unhappy, mildly frustrated
            - frustrated: strong frustration, anger, harsh language, threatening tone

            Rules:
            - Pick only one category
            - Focus on the overall emotion, not individual words
            - Do not answer the complaint or give suggestions
            """
        )
    ]
    output = str_sentiment.invoke(message)
    return {"sentiment": output.sentiment}

def draft_response(state:complain_state):
    complain = state["complain"]
    severity = state["severity"]
    sentiment = state["sentiment"]
    category = state["category"]
    message = [
        SystemMessage(
            content="You are a professional, empathetic customer-support reply writer. You craft clear, polite, solution-focused replies tailored to the customer's category, severity, and sentiment."
        ),
        HumanMessage(
            content=f"""
            Write a customer-support reply based on these details:
            Customer complaint: {complain}
            Order ID: {state['order_id']}
            Category: {category}
            Sentiment: {sentiment}
            Severity: {severity}

            Tone guidance:
            - frustrated or disappointed → extra apologetic, calm, reassuring
            - neutral → clear, direct, solution-focused
            - happy → warm, appreciative

            Urgency guidance:
            - Critical or High → show clear urgency and immediate next steps
            - Medium or Low → give realistic, polite timelines

            Requirements:
            - Stay professional, polite, and empathetic at all times
            - Keep it under 120 words
            - No false promises
            - No unnecessary questions
            - Start by acknowledging the issue; close helpfully
            """
        )
    ]
    output = model.invoke(message).content
    return {"draft_response": output}


def evaluator_draft(state:complain_state):
    draft_response = state["draft_response"]
    message = [
        SystemMessage(
            content="You are a strict but fair quality evaluator for customer-support draft replies. Verify the reply is professional and matches the required Category, Sentiment, and Severity, then judge overall quality."
        ),
        HumanMessage(
            content=f"""
            Evaluate this draft reply:
            {draft_response}

            Customer complaint: {state['complain']}

            Criteria:
            - Tone Match: does the tone match the Sentiment? (frustrated/disappointed → apologetic/calm/reassuring; neutral → clear/professional; happy → warm/positive)
            - Severity Handling: does the urgency match? (Critical/High → urgent + immediate next steps; Medium/Low → realistic/polite)
            - Category Relevance: does it address the main Category (Payment / Delivery / Product / Other)?

            Quality Checks:
            - Polite and professional language
            - No false promises
            - Concise (under 120 words)
            - No unnecessary questions
            - Helpful closing

            Output exactly ONE verdict: Approved or Need_improved.
            Then provide concise constructive feedback.
            """
        ),
        HumanMessage(
            content=f"""
            Provide the verdict and feedback for the draft above. Verdict must be exactly 'Approved' or 'Need_improved'.
            """
        )
    ]
    output = str_evaluator.invoke(message)
    return {"evaluator":output.evaluator , "feedback":output.feedback}

def optimizer_draft(state:complain_state):
    feedback = state["feedback"]
    evaluator = state["evaluator"]
    message = [
        SystemMessage(
            content="You are an expert customer-support reply optimizer. Improve a draft that was marked 'Need_improved', fixing exactly the issues the evaluator flagged while preserving the original intent."
        ),
        HumanMessage(
            content=f"""
            Optimize the draft reply below using the evaluator's feedback.

            Customer complaint: {state['complain']}
            Category: {state['category']}
            Sentiment: {state['sentiment']}
            Severity: {state['severity']}
            Current draft reply: {state['draft_response']}
            Evaluator feedback: {feedback}

            Rules:
            - Fix only the issues mentioned in the Evaluator Feedback
            - Keep the tone matched to Sentiment (frustrated/disappointed → apologetic, calm, reassuring; neutral → clear, professional; happy → warm, positive)
            - Adjust urgency to Severity (Critical/High → urgent + next steps; Medium/Low → realistic, polite)
            - Properly address the Category issue
            - Keep it concise (under 120 words)
            - No false promises, no unnecessary questions
            - End with a professional, helpful closing
            """
        )
    ]
    output = model.invoke(message).content
    iteration = state["iteration"] +1
    return {"draft_response":output , "iteration":iteration}



def evaluator_decision(state:complain_state):
    # Force at least one optimizer round first (iteration 1 must go through optimizer)
    if state["iteration"] >= state["max_iteration"]:
        return "Approved"
    if state["iteration"] <= 1:
        return "Need_improved"
    if state["evaluator"] == "Approved":
        return "Approved"
    return "Need_improved"
    
def final(state:complain_state):
    if state["severity"] == "Critical" or state["sentiment"] == "frustrated":
        res = {"escalate_to_human":"Sent to Human"}
        print(res)
    else:
        res = {"auto_send":"Auto send to Customer"}
        print(res)

graph = StateGraph(complain_state)

graph.add_node("categorize_complain" , categorize_complain)
graph.add_node("severity_check" , severity_check)
graph.add_node("sentiment_check" , sentiment_check)
graph.add_node("draft_response" , draft_response)
graph.add_node("evaluator_draft" , evaluator_draft)
graph.add_node("optimizer_draft" , optimizer_draft)
graph.add_node("final" , final)


graph.add_edge(START , "categorize_complain")
graph.add_edge("categorize_complain" , "severity_check")
graph.add_edge("categorize_complain" , "sentiment_check")
graph.add_edge("severity_check" , "draft_response")
graph.add_edge("sentiment_check" , "draft_response")
graph.add_edge("draft_response" , "evaluator_draft")
graph.add_conditional_edges("evaluator_draft" , evaluator_decision, {"Approved":"final" , "Need_improved":"optimizer_draft"})
graph.add_edge("optimizer_draft" , "evaluator_draft")

graph.add_edge("final",END)


workflow = graph.compile()

inital_state = {
    "complain":"I ordered a wireless earphone from your website on 14 april 2026. The payment was successfully made via JazzCash, but my order has still not been delivered. The tracking status only shows “Processing.” When can I expect to receive my order? There is a significant delay.",
    "order_id": "#PK78432",
    "max_iteration":3,
    "iteration":1
}

response = workflow.invoke(inital_state)
print(response["category"])
print(response["severity"])
print(response["sentiment"])
print(response["draft_response"])
print(response["evaluator"])
print(f"FeedBack:{response["feedback"]}")
print(response["iteration"])