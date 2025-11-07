from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import models, transforms
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

model = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
model.eval()

dummy_inputs = torch.randn(1,3,224,224)
torch.onnx.export(model, dummy_inputs, "07_model.onnx")

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/predict', methods=['POST'])
def predict():
    print(request.files)
    if 'file' not in request.files:
        return jsonify({'error': 'No File part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    print(file.filename)

    # doe een voorspelling
    try:
        image = Image.open(io.BytesIO(file.read())) # lees de figuur
        image = preprocess(image)
        image = image.unsqueeze(0) # 3, B, H -> 1, 3, B, H
        print(image.shape)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            result = jsonify({'class': predicted.item()})
            print(result)
            return result, 200
    except Exception as e:
        return jsonify({'error': str(e)}, 500)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
