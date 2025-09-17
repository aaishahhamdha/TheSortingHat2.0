import pandas as pd
from catboost import CatBoostClassifier, CatboostError, Pool
import random

from .generator import generate_message

# Load the model
model = CatBoostClassifier()
try:
    model.load_model("the_sorting_hat_2.0_model.cbm")
except FileNotFoundError:
    print("Model file not found. Make sure the path is correct.")
except CatboostError as e:
    print(f"Error loading model: {e}")

threshold=0.15

# Sorting Hat logic
def predict_house(student_data: object):
    required_attrs = ["blood_status", "bravery", "intelligence", "loyalty",
                  "ambition", "dark_arts", "quidditch", "dueling", "creativity"]
    valid_blood_status = ["Pure-blood", "Half-blood", "Muggle-born"]

    for attr in required_attrs:
        if not hasattr(student_data, attr):
            raise ValueError(f"Missing required student attribute: {attr}")
    
    if student_data.blood_status not in valid_blood_status:
        raise ValueError(f"Invalid Blood Status: {student_data.blood_status}")
        
    student = pd.DataFrame([{
        "Blood Status": student_data.blood_status,
        "Bravery": student_data.bravery,
        "Intelligence": student_data.intelligence,
        "Loyalty": student_data.loyalty,
        "Ambition": student_data.ambition,
        "Dark Arts Knowledge": student_data.dark_arts,
        "Quidditch Skills": student_data.quidditch,
        "Dueling Skills": student_data.dueling,
        "Creativity": student_data.creativity
    }])
    cat_features = ["Blood Status"]
    student_pool = Pool(data=student, cat_features=cat_features)

    try:
        predicted_house = model.predict(student_pool).item()
        probabilities = model.predict_proba(student_pool)[0]
    except Exception as e:
        raise RuntimeError(f"Prediction failed: {e}")

    prob_dict = dict(zip(model.classes_, probabilities))

    student_info = str(student_data)
    prob_info = str(prob_dict)
    message_hat = generate_message(
    "You are the famous Harry Potter Sorting Hat. Write a personalised message for this student. "
    "Do not reveal any raw scores or probability values. "
    "Keep the response within 100-120 words. "
    "Use conversational fillers like 'ahh', 'mmm', or dramatic expressions. "
    "If the student has strong traits from multiple houses, mention that it was difficult to decide, "
    "and explain how their personality shows qualities from more than one house. "
    f"Student data: {student_info} "
    f"Predicted house: {predicted_house} "
    f"Predicted probabilities: {prob_info}"
    )

    message_doc = generate_message(
        f"You are a result generator who explains in simple and clear language why a student was placed into a specific house based on their personality traits. "
        f"Start with 'You belong in {predicted_house}.' or 'Your house is {predicted_house}.' "
        "Do not use conversational fillers like 'ahh', 'mmm', or dramatic expressions. "
        "Do not reveal any raw scores or probability values. "
        "Keep the response within 250 words. "
        "Use plain, age-friendly wording that children can understand, while keeping the style like a short evaluation report. "
        "If the student has strong traits from multiple houses, mention that it was difficult to decide, "
        "and explain how their personality shows qualities from more than one house. "
        f"Hats thinking and sorting: {message_hat} "
        f"Student data: {student_info} "
        f"Predicted house: {predicted_house} "
        f"Predicted probabilities: {prob_info}"
    )

    return predicted_house, prob_dict, message_hat, message_doc
