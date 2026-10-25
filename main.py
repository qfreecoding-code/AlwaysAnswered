import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import urllib.request
import urllib.error

app = FastAPI(title="AlwaysAnswered Neural Engine", redirect_slashes=False)

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
    return {"status": "AlwaysAnswered Neural Engine is online!"}

@app.post("/api/chat")
def process_chat(inquiry: ChatInquiry):
    # 1. Load client profile JSON dynamically from your repository's clients/ folder
    profile_path = os.path.join("clients", f"{inquiry.client_id}.json")
    if not os.path.exists(profile_path):
        raise HTTPException(status_code=404, detail="Client profile not found")
        
    with open(profile_path, "r") as f:
        client_profile = json.load(f)
        
    biz = client_profile['business_info']
    
    # 2. Rich System Prompt instructing the neural net to speak like a professional human receptionist
    system_instruction = f"""
    You are an expert, friendly AI receptionist for {biz['business_name']}, a professional {biz['industry']} company.
    Operating Hours: {biz['operating_hours']}
    
    Rate Card & Pricing Rules:
    {json.dumps(client_profile['rate_card'], indent=2)}
    
    Your job:
    - Greet the customer warmly and naturally (no robotic pre-written templates).
    - Answer questions about availability or pricing accurately using the rate card.
    - Politely guide them to provide their Name, Phone Number, Service Address, and Issue Details so a technician can be dispatched.
    - Return your final response in clean JSON format with keys: "reply_message", "intent_detected", and "lead_collected".
    """

    # 3. Neural inference request routed to a high-speed cloud neural endpoint
    api_key = os.environ.get("GEMINI_API_KEY")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    
    payload = {
        "contents": [{"parts": [{"text": inquiry.message}]}],
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "temperature": 0.3,  # Slight temperature for natural, flowing responses
            "maxOutputTokens": 400
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
            
            # Clean markdown wrappers if returned by the neural net
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