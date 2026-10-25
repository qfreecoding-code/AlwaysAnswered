import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import urllib.request
import urllib.error

app = FastAPI(title="AlwaysAnswered Micro Engine", redirect_slashes=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatInquiry(BaseModel):
    client_id: str
    message: str

@app.get("/")
def read_root():
    return {"status": "AlwaysAnswered Micro Engine is online!"}

@app.post("/api/chat")
def process_chat(inquiry: ChatInquiry):
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

    # 3. Fast direct HTTP REST payload to Gemini Flash (using active gemini-3.8-flash endpoint)
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY environment variable is not set on the server.")
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"parts": [{"text": inquiry.message}]}],
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.2,
            "maxOutputTokens": 300
        }
    }
    
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            res_body = json.loads(response.read().decode("utf-8"))
            raw_text = res_body["candidates"][0]["content"]["parts"][0]["text"].strip()
            
            # Clean markdown code blocks if present
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            elif raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
                
            return json.loads(raw_text.strip())
            
    except urllib.error.HTTPError as e:
        error_message = e.read().decode("utf-8")
        raise HTTPException(status_code=e.code, detail=error_message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)