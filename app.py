# app.py
from flask import Flask, request, jsonify, send_file
from tasks import kickoff_pipeline
from rag_integration import ingest_notes

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_notes():
    file = request.files['notes']
    path = f"./uploads/{file.filename}"
    file.save(path)
    ingest_notes(path)
    return jsonify({"status": "indexed notes"})

@app.route('/summarize', methods=['POST'])
def summarize():
    query = request.json.get('query')
    ground_truth = request.json.get('ground_truth', '')
    results = kickoff_pipeline(query, ground_truth)
    return jsonify({"results": results})

if __name__ == '__main__':
    app.run(port=8000)
