from flask import Flask, request, jsonify
from EmotionDetection import emotion_detector

app = Flask(__name__)


@app.route('/emotionDetector', methods=['POST'])
def emotion_detector_endpoint():
    """
    Endpoint to analyze emotions in the given text.
    Returns:
        JSON: Formatted response containing emotion scores and dominant emotion,
              or an error message for invalid input.
    """
    # Extract the text to analyze from the POST request
    text_to_analyze = request.json.get('text')

    if not text_to_analyze or not text_to_analyze.strip():
        # Handle blank input
        return jsonify({"error": "Invalid text! Please try again!"}), 400

    # Call the emotion detector function
    result = emotion_detector(text_to_analyze)

    if 'error' in result:
        # Handle errors in the emotion detector function
        return jsonify(result), 500

    if result['dominant_emotion'] == 'none':
        # Handle cases where no dominant emotion is detected
        return jsonify({"error": "Invalid text! Please try again!"}), 400

    # Format the response message
    response_message = (
        f"For the given statement, the system response is "
        f"'anger': {result['anger']}, "
        f"'disgust': {result['disgust']}, "
        f"'fear': {result['fear']}, "
        f"'joy': {result['joy']} and "
        f"'sadness': {result['sadness']}. "
        f"The dominant emotion is {result['dominant_emotion']}."
    )

    # Return the formatted response
    return jsonify({"message": response_message})


if __name__ == '__main__':
    # Run the Flask server on localhost, port 5000
    app.run(debug=True, host='0.0.0.0', port=5001)
