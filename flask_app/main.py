from flask import Flask, request, jsonify
from onnx_classifier import OnnxClassifier
import numpy as np 
app = Flask(__name__)


classifier = OnnxClassifier(model_path="/workspace/models/onnx-model-gpu/1/model.onnx")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.json
        input = np.array(data['input'])
        output = classifier.infer(input.astype(np.float32))
        return jsonify({'output': output.tolist()})


if __name__== '__main__':
    app.run(debug=True, host='0.0.0.0')
