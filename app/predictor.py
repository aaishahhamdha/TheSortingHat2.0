import pandas as pd
from catboost import CatBoostClassifier, Pool
import random

# Load the model
model = CatBoostClassifier()
try:
    model.load_model("the_sorting_hat_2.0_model.cbm")
except FileNotFoundError:
    print("Model file not found. Make sure the path is correct.")
except CatBoostError as e:
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

    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    top_house, top_prob = sorted_probs[0]
    second_house, second_prob = sorted_probs[1]

    close_message = ""
    if top_prob - second_prob < threshold:
        close_message = "You have multiple strong traits that made it difficult to place you in one house, " \
                        "but a Sorting Hat never fails! "

    house_emoji = {"Gryffindor": "🦁", "Hufflepuff": "🦡", "Ravenclaw": "🦅", "Slytherin": "🐍"}
    house_messages = {
        "Gryffindor": ["Your courage and daring spirit make you a true Gryffindor!"],
        "Hufflepuff": ["Your loyalty and dedication shine bright in Hufflepuff!"],
        "Ravenclaw": ["Your wisdom and creativity belong in Ravenclaw!"],
        "Slytherin": ["Your ambition and cunning mark you as a true Slytherin!"]
    }

    selected_message = random.choice(house_messages[predicted_house])

    house_traits = {
        "Gryffindor": ["Bravery", "Ambition"],
        "Hufflepuff": ["Loyalty", "Bravery"],
        "Ravenclaw": ["Intelligence", "Creativity"],
        "Slytherin": ["Ambition", "Dark Arts Knowledge"]
    }
    top_traits = house_traits[predicted_house]

    message = f"🎓 {student_data.name}, welcome to {predicted_house}! {house_emoji[predicted_house]}\n\n"
    message += f"The Sorting Hat has spoken! You are exceptional in {top_traits[0].lower()} and {top_traits[1].lower()}. "
    message += "But remember, your house placement considers all your traits — bravery, intelligence, loyalty, ambition, dark arts knowledge, and more. "
    message += close_message + selected_message

    return predicted_house, prob_dict, message
