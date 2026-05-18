#!/usr/bin/env python3
"""
Mistral 2B Chatbot Backend Server
Runs on your Mac, communicates with your iPhone over local WiFi
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import ollama
import logging

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "mistral:2b"
SERVER_PORT = 5000

# Initialize Ollama client
client = ollama.Client()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    try:
        # Quick test to see if model is available
        client.list()
        return jsonify({'status': 'healthy', 'model': MODEL_NAME}), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        character_prompt = data.get('character_prompt', '')

        if not user_message:
            return jsonify({'error': 'Empty message'}), 400

        logger.info(f"User: {user_message}")

        # Build the full prompt with character context
        full_prompt = f"{character_prompt}\n\nUser: {user_message}\n\nAssistant:"

        # Call Mistral 2B via Ollama
        response = client.generate(
            model=MODEL_NAME,
            prompt=full_prompt,
            stream=False,
            options={
                'temperature': 0.7,
                'top_p': 0.9,
                'top_k': 40,
            }
        )

        assistant_message = response['response'].strip()
        logger.info(f"Assistant: {assistant_message}")

        return jsonify({
            'response': assistant_message,
            'model': MODEL_NAME
        }), 200

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    try:
        models = client.list()
        return jsonify({'models': models}), 200
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info(f"Starting Mistral 2B Chatbot Server on port {SERVER_PORT}")
    logger.info(f"Make sure Ollama is running with: ollama serve")
    logger.info(f"Pull Mistral 2B with: ollama pull mistral:2b")
    logger.info(f"\nServer will be accessible at: http://localhost:{SERVER_PORT}")
    logger.info("On iPhone, use your Mac's local IP (find with 'ifconfig')")
    
    app.run(
        host='0.0.0.0',  # Listen on all interfaces (local network)
        port=SERVER_PORT,
        debug=False,
        use_reloader=False
    )
