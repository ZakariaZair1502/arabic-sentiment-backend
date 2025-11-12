from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import requests
import re
import torch
import torch_directml

app = Flask(__name__)
CORS(app)

# --- Initialize DirectML device ---
try:
    if torch_directml.is_available():
        dml_device = torch_directml.device()
        print(f"DirectML device detected: {dml_device}. Using DirectML for inference.")
    else:
        dml_device = "cpu"
        print("DirectML not available. Falling back to CPU for inference.")
except Exception as e:
    dml_device = "cpu"
    print(f"Error initializing DirectML: {e}. Falling back to CPU for inference.")

# --- Load the Darija Sentiment Analysis Model ---
tokenizer = None
model = None
sentiment_analyzer = None
id_to_label = {}

try:
    model_name = "BenhamdaneNawfal/sentiment-analysis-darija"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    if dml_device != "cpu":
        model = model.to(dml_device)
        print(f"Model moved to DirectML device: {dml_device}")
    else:
        print("Model staying on CPU as DirectML is not available.")

    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model=model,
        tokenizer=tokenizer
    )# type: ignore
    print(f"Darija Sentiment Analysis model '{model_name}' loaded successfully!")
    
    id_to_label = model.config.id2label
    print(f"Model ID to Label mapping: {id_to_label}")

except Exception as e:
    print(f"Error loading sentiment model: {e}")
    sentiment_analyzer = None
    id_to_label = {}

# --- Facebook Comment Extraction Function (remains mostly the same) ---
def extract_facebook_comments(post_id, page_access_token):
    all_comments = []
    url = f"https://graph.facebook.com/v24.0/{post_id}/comments"
    params = {
        "access_token": page_access_token,
        "limit": 100  # maximum per page
    }

    while url:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            print("Error:", response.json())
            break

        data = response.json()
        for comment in data.get('data', []):
            all_comments.append(comment.get('message', ''))

        # Paging
        url = data.get('paging', {}).get('next')

    return all_comments

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    if sentiment_analyzer is None:
        return jsonify({"error": "Sentiment analysis model not loaded."}), 500

    data = request.json or {}
    post_id_input = data.get('post_id') # Renamed from 'input' to 'post_id'
    access_token = data.get('accessToken')

    if not post_id_input:
        return jsonify({"error": "No Facebook Post ID provided"}), 400
    if not access_token:
        return jsonify({"error": "Access Token is required to fetch comments from a Facebook Post ID."}), 400

    resolved_post_id = None

    # Validate input is either page_id_post_id or numerical post ID
    if re.match(r"^\d+_\d+$", post_id_input.strip()): # Check for page_id_post_id
        resolved_post_id = post_id_input.strip()
        print(f"Input treated as direct Facebook combined post ID: {resolved_post_id}")
    elif re.match(r"^\d+$", post_id_input.strip()): # Check for numerical ID
        resolved_post_id = post_id_input.strip()
        print(f"Input treated as direct numerical Facebook post ID: {resolved_post_id}")
    else:
        return jsonify({"error": "Invalid Facebook Post ID format. Please provide a numerical ID or a page_id_post_id (e.g., 123_456)."}), 400

    # If we successfully got a post ID and have an access token, proceed
    if resolved_post_id and access_token:
        print(f"Proceeding to extract comments for post ID: {resolved_post_id}")
        extracted_comments = extract_facebook_comments(resolved_post_id, access_token)
        
        if extracted_comments:
            comments_to_analyze = extracted_comments # Use the extracted comments
        else:
            return jsonify({"error": f"No comments found for post ID {resolved_post_id}. This might be due to incorrect Access Token permissions, the post having no comments, or the post not being publicly accessible. Check the Graph API Explorer with this post ID and your token to confirm."}), 400
    else: # Should not be reached with the checks above, but as a safeguard
        return jsonify({"error": "An unexpected error occurred during post ID validation or token check."}), 500


    results = []
    if not comments_to_analyze:
        return jsonify({"error": "No comments found to analyze. Please check your input."}), 400

    batch_size = 16 
    for i in range(0, len(comments_to_analyze), batch_size):
        batch_comments = comments_to_analyze[i:i + batch_size]
        
        try:
            sentiment_outputs = sentiment_analyzer(batch_comments)
            
            for j, output in enumerate(sentiment_outputs):
                original_comment_index = i + j
                comment_text = comments_to_analyze[original_comment_index]
                
                final_sentiment = output['label'].lower() 
                if final_sentiment not in ["positive", "negative"]:
                    if output['label'] == id_to_label.get(0, 'LABEL_0'):
                        final_sentiment = "negative"
                    elif output['label'] == id_to_label.get(1, 'LABEL_1'):
                        final_sentiment = "positive"
                    else:
                        final_sentiment = "negative" 

                results.append({
                    "id": str(original_comment_index + 1),
                    "text": comment_text,
                    "sentiment": final_sentiment,
                    "confidence": float(output['score'])
                })
        except Exception as e:
            print(f"Error processing a batch of comments: {e}")
            for j in range(len(batch_comments)):
                original_comment_index = i + j
                comment_text = comments_to_analyze[original_comment_index]
                results.append({
                    "id": str(original_comment_index + 1),
                    "text": comment_text,
                    "sentiment": "negative",
                    "confidence": 0.5
                })

    return jsonify(results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
