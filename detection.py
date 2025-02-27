import spacy
from transformers import pipeline
from collections import defaultdict

# Load the spaCy model for NER
nlp = spacy.load("en_core_web_sm")  # python -m spacy download en_core_web_sm

# Load a pre-trained intent classification model from Hugging Face
intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define the available intents
available_intents = ["BookFlight", "CheckWeather", "OrderFood"]

# Simulating different agents
agents = {
    "BookFlight": "Flight Booking Agent",
    "CheckWeather": "Weather Agent",
    "OrderFood": "Food Delivery Agent"
}

# Function for Intent Classification
def classify_intent(text):
    result = intent_classifier(text, candidate_labels=available_intents)
    return result['labels'][0]  # Return the top intent

# Function for Named Entity Recognition (NER)
def extract_entities(text):
    doc = nlp(text)
    entities = defaultdict(list)
    for ent in doc.ents:
        entities[ent.label_].append(ent.text)
    return entities

# Main Function to process user input
def process_user_input(text):
    print(f"User Input: {text}")

    # Step 1: Identify the intent using zero-shot classification
    intent = classify_intent(text)
    print(f"Identified Intent: {intent}")

    # Step 2: Extract entities (e.g., dates, locations)
    entities = extract_entities(text)
    print(f"Extracted Entities: {dict(entities)}")

    # Step 3: Route to the appropriate agent based on the intent
    agent = agents.get(intent, "Default Agent")
    print(f"Routing to Agent: {agent}")

    # Returning all information for the sake of the example
    return {
        "intent": intent,
        "entities": dict(entities),
        "agent": agent
    }

# Example User Inputs
user_inputs = [
    "I want to book a flight to New York for tomorrow.",
    "What's the weather in Los Angeles?",
    "Can you order a pizza for me?",
    "I want to book return flight to India for 03-March-2025",
    "I ordered from my fav hotel. Do you want to join me?"
]

# Process each user input
for user_input in user_inputs:
    process_user_input(user_input)
    print("\n---\n")
