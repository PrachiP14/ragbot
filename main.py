from flask import Flask, render_template, request, jsonify
from rag import RAGSystem
import os

app = Flask(__name__)

# Fixed PDF path and API key
PDF_PATH = "Collage.pdf"
API_KEY = "AIzaSyBVbQcYILiH1-JUH120u9z48_4ZHWrgftE"

# Initialize RAG system
rag = None

def initialize_rag():
    global rag
    if rag is None:
        rag = RAGSystem(PDF_PATH, API_KEY)
    return rag

@app.route('/')
def index():
    return render_template('landing.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/features')
def features():
    return render_template('features.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/chat')
def chat():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.json
    question = data.get('question', '')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        rag_system = initialize_rag()
        answer = rag_system.get_answer(question)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '_main_':
    app.run(debug=True)