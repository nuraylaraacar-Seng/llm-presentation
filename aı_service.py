from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import requests
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:5500",
        "null"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


@app.post("/chat")
def chat(payload: dict):
    prompt = payload.get("prompt", "")
    context = payload.get("context", "")

    if not OPENROUTER_API_KEY:
        return {"response": "OPENROUTER_API_KEY missing"}

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "meta-llama/llama-3.1-8b-instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": f"You are a presentation AI assistant.\nContext: {context}"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        )

        data = response.json()

        return {
            "response": data["choices"][0]["message"]["content"]
        }

    except Exception as e:
        return {"response": f"Error: {str(e)}"}