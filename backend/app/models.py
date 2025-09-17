from pydantic import BaseModel

class Student(BaseModel):
    name: str
    blood_status: str
    bravery: float
    intelligence: float
    loyalty: float
    ambition: float
    dark_arts: float
    quidditch: float
    dueling: float
    creativity: float

class PredictionResponse(BaseModel):
    name: str
    predicted_house: str
    house_probabilities: dict
    message_hat: str
    message_doc: str