import os
import json
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="AlwaysAnswered Lightning Engine", redirect_slashes=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the Gemini client once on startup to keep the connection warm and eliminate latency
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None

class ChatInquiry(BaseModel):
    client_id: str
    message: str

@app.get("/")
def read_root():
    return {"status": "AlwaysAnswered Lightning Engine is online!"}

@app.post("/api/chat")
def process_chat(inquiry: ChatInquiry):
    if not client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is not set on the server.")
        
    try:
        # 1. Load client profile JSON
        profile_path = os.path.join("clients", f"{inquiry.client_id}.json")
        if not os.path.exists(profile_path):
            raise HTTPException(status_code=404, detail="Client profile not found")
            
        with open(profile_path, "r") as f:
            client_profile = json.load(f)
            
        biz = client_profile['business_info']
        
        # 2. Build system instructions for the AI receptionist
        system_instruction = f"""
        You are an expert, friendly AI receptionist for {biz['business_name']}, a professional {biz['industry']} company.
        Operating Hours: {biz['operating_hours']}
        
        Rate Card & Pricing Rules:
        {json.dumps(client_profile['rate_card'], indent=2)}
        
        Greet warmly, answer pricing or availability questions using the rate card, and collect lead info (Name, Phone, Address, Issue). Return your final response in valid JSON format.
        """

        # 3. Blazing fast official SDK call with pre-warmed connection pool
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=inquiry.message,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                temperature=0.2,
                maxOutputTokens=300
            ),
        )
        
        raw_text = response.text.strip()
        
        # Clean markdown code blocks if present
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        elif raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
            
        return json.loads(raw_text.strip())
        
    except Exception as e:
        print("--- CRITICAL EXCEPTION TRACEBACK ---")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)