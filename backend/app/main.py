from fastapi import FastAPI
from app.models import PredictionResponse, Student
from app.predictor import predict_house

app=FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to the Hogwarts House Predictor API"}

@app.post("/sort",response_model=PredictionResponse)
def sort(student: Student):
    predicted_house,house_probabilities,message = predict_house(student)
    return {
        "name": student.name,
        "predicted_house": predicted_house,
        "house_probabilities": house_probabilities,
        "message": message
    }
