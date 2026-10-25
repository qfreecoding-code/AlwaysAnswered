import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

app = FastAPI(title="AlwaysAnswered Neural Engine", redirect_slashes=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load a true lightweight neural net directly into server memory on startup
# Using a fast, highly capable small model (e.g., Qwen2.5-1.5B-Instruct or Phi-3-mini)
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

print("Loading onboard neural net into memory...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32, # CPU-compatible precision for Render free tier
    low_cpu_mem_usage=True
)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
print("Neural net loaded successfully!")

class ChatInquiry(BaseModel):
    client_id: str
    message: str

@app.get("/")
def read_root():
    return {"status": "AlwaysAnswered Neural Engine is online and running local weights!"}

@app.post("/api/chat")
def process_chat(inquiry: ChatInquiry):
    profile_path = os.path.join("clients", f"{inquiry.client_id}.json")
    if not os.path.exists(profile_path):
        raise HTTPException(status_code=404, detail="Client profile not found")
        
    with open(profile_path, "r") as f:
        client_profile = json.load(f)
        
    system_prompt = f"""You are an AI receptionist for {client_profile['business_info']['business_name']}.
Industry: {client_profile['business_info']['industry']}.
Hours: {client_profile['business_info']['operating_hours']}.
Rate Card: {json.dumps(client_profile['rate_card'])}.
Greet the customer naturally, answer their question using the rate card, and ask for their contact details and address. Speak like a real human trade receptionist."""

    # Format prompt for instruction-tuned neural model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": inquiry.message}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Generate true neural net tokens
    outputs = generator(
        formatted_prompt,
        max_new_tokens=150,
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    
    generated_text = outputs[0]['generated_text']
    # Extract the assistant's reply after the prompt tokens
    response_reply = generated_text[len(formatted_prompt):].strip()

    return {
        "business_name": client_profile['business_info']['business_name'],
        "ai_engine": MODEL_NAME,
        "reply_message": response_reply,
        "lead_info_collected": {
            "customer_message": inquiry.message,
            "status": "processed_by_local_neural_net"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)