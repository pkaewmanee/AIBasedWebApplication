import requests

def emotion_detector(text_to_analyze):
    """
    Detect emotions in a given text using Watson NLP API and return a dictionary
    containing the scores for anger, disgust, fear, joy, and sadness, along 
    with the dominant emotion.
    
    Args:
        text_to_analyze (str): The text to analyze.

    Returns:
        dict: A dictionary with the following structure:
        {
            'anger': anger_score,
            'disgust': disgust_score,
            'fear': fear_score,
            'joy': joy_score,
            'sadness': sadness_score,
            'dominant_emotion': '<name of the dominant emotion>'
        }
        or an error dictionary if something went wrong.
    """
    # Check for blank text input
    if not text_to_analyze.strip():
        return {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': 'none'
        }

    # Watson NLP API URL and headers
    url = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
    headers = {
        "grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock",
        "Content-Type": "application/json"
    }

    # Prepare the payload
    payload = {
        "raw_document": {
            "text": text_to_analyze
        }
    }

    try:
        # Make the POST request
        response = requests.post(url, json=payload, headers=headers)

        # Check if the request was successful
        if response.status_code == 200:
            response_data = response.json()

            # Extract emotions from the correct key in the response
            try:
                emotions = response_data["emotionPredictions"][0]["emotion"]
                anger_score = emotions.get("anger", 0.0)
                disgust_score = emotions.get("disgust", 0.0)
                fear_score = emotions.get("fear", 0.0)
                joy_score = emotions.get("joy", 0.0)
                sadness_score = emotions.get("sadness", 0.0)
                
                # Prepare the emotion scores dictionary
                scores_dict = {
                    'anger': anger_score,
                    'disgust': disgust_score,
                    'fear': fear_score,
                    'joy': joy_score,
                    'sadness': sadness_score
                }

                # Find the dominant emotion
                dominant_emotion = max(scores_dict, key=scores_dict.get)

                # Create the final result with dominant emotion
                result = {
                    'anger': anger_score,
                    'disgust': disgust_score,
                    'fear': fear_score,
                    'joy': joy_score,
                    'sadness': sadness_score,
                    'dominant_emotion': dominant_emotion
                }

                return result
            
            except (KeyError, IndexError, TypeError) as e:
                return {
                    "error": "Unexpected response format from the Watson NLP API",
                    "details": str(e),
                    "raw_response": response_data
                }
        elif response.status_code == 400:
            # If the status code is 400, return None for emotions
            return {
                'anger': None,
                'disgust': None,
                'fear': None,
                'joy': None,
                'sadness': None,
                'dominant_emotion': 'none'
            }
        else:
            # Handle unexpected status codes
            return {
                "error": f"Unexpected status code {response.status_code}",
                "details": response.text
            }
    except requests.exceptions.RequestException as e:
        # Handle request errors
        return {"error": str(e)}
