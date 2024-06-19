from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
from openai import OpenAI

import os
from dotenv import load_dotenv

load_dotenv()
oai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=oai_api_key)


app = FastAPI()


classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


candidate_labels = [
    "shutdown computer",
    "turn on computer",
    "internet not working on the computer",
    "computer restarts repeatedly",
    "other",
]


class IntentRequest(BaseModel):
    text: str


class IntentRecognizer:
    def __init__(self):
        self.classifier = classifier

    def recognize_intent(self, text):
        result = self.classifier(text, candidate_labels)
        return result["labels"][0]


intent_recognizer = IntentRecognizer()


@app.post("/recognize-intent")
def recognize_intent(request: IntentRequest):
    try:
        recognized_intent = intent_recognizer.recognize_intent(request.text)

        if recognized_intent in candidate_labels and recognized_intent != "other":
            openai_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": f"This is the user's message: '{request.text}'. This is the recognized intent: '{recognized_intent}'. Can you help the user?",
                    },
                ],
                max_tokens=150,
            )
            openai_message = openai_response.choices[0].message.content.strip()
            return {
                "recognized_intent": recognized_intent,
                "message": openai_message,
            }
        else:
            return {
                "recognized_intent": recognized_intent,
                "message": "I am not able to help with that.",
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
