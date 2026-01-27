import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from agent import agent_executor 

load_dotenv()
app = FastAPI(title="Accounting AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, change this to your Vercel URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

# 5.Routes
@app.get("/")
def health_check():
    return {"status": "Agent Standing By"}

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        # Debug: Print what we received
        print(f"Received {len(req.messages)} messages from frontend.")

        # Convert Frontend Messages to LangChain Format
        langchain_messages = []
        for msg in req.messages:
            if msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "ai":
                langchain_messages.append(AIMessage(content=msg.content))

        # Run the Agent with History
        inputs = {"messages": langchain_messages}
        result = agent_executor.invoke(
            inputs,
            config={"recursion_limit": 5}
        )

        # Extract the Final Answer
        # The result may be a dict with 'messages' key
        messages = result.get("messages", [])
        if not messages:
            raise HTTPException(status_code=500, detail="Agent returned no messages")
        
        last_message = messages[-1]
        
        # Extract content from last message
        raw_content = None
        if hasattr(last_message, 'content'):
            raw_content = last_message.content
        elif isinstance(last_message, dict):
            raw_content = last_message.get('content')
        else:
            raw_content = str(last_message)
        
        # CHECK: Is it a simple string or a list of blocks?
        if isinstance(raw_content, str):
            final_answer = raw_content
        elif isinstance(raw_content, list):
            # It's a list like [{"type": "text", "text": "..."}]
            # We join all the text parts together
            final_answer = "".join(
                [block.get("text", "") for block in raw_content if isinstance(block, dict) and "text" in block]
            )
        else:
            # Fallback for unknown formats
            final_answer = str(raw_content)

        print("Response sent.")
        return {"answer": final_answer}
    
    except Exception as e:
        print(f"Error: {e}")
        # Print the line number
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
