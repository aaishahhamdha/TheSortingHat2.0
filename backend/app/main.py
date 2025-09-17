from fastapi import FastAPI
from app.models import PredictionResponse, Student
from app.predictor import predict_house
from fastapi.middleware.cors import CORSMiddleware

app=FastAPI()

origins = [
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Welcome to the Hogwarts House Predictor API"}

@app.post("/sort",response_model=PredictionResponse)
def sort(student: Student):
    predicted_house, house_probabilities, message_hat, message_doc = predict_house(student)
    return {
        "name": student.name,
        "predicted_house": predicted_house,
        "house_probabilities": house_probabilities,
        "message_hat": message_hat,
        "message_doc":message_doc
    }
