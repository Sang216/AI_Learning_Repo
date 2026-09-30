# memory_demo.py
# Block 10 — Version 1: memory with the history typed by hand
# (shows the IDEA: the model only "remembers" because WE re-send past messages).
# Run:  python memory_demo.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly tutor."),
    MessagesPlaceholder("history"),     # past turns slot in here
    ("human", "{question}"),
])
chain = prompt | model

# we hardcode the history here just to demonstrate
history = [HumanMessage("My name is Aarav."), AIMessage("Hi Aarav!")]

answer = chain.invoke({"history": history, "question": "What's my name?"})
print(answer.content)   # -> "Your name is Aarav."  (it "remembered" because we re-sent history)
