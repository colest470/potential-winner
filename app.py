import base64
import io
from flask import Flask, render_template, request
from PIL import Image
import src.network5 as network5

app = Flask(__name__)

try:
    model = network5.load_trained_model("my_trained_network5.pth")
    scaler = network5.build_scaler()
except FileNotFoundError:
    model = None
    scaler = None

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_data = None
    error = None

    if request.method == 'POST':
        uploaded_file = request.files.get('image')
        if not uploaded_file:
            error = 'No image uploaded.'
        elif model is None:
            error = 'Model not loaded. Please start the app after placing my_trained_network5.pth in the project root.'
        else:
            try:
                image = Image.open(uploaded_file.stream)
                prediction = network5.predict_image(image, model, scaler=scaler)

                buffer = io.BytesIO()
                image.convert('RGB').save(buffer, format='PNG')
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            except Exception as exc:
                error = f'Unable to process image: {exc}'

    return render_template('index.html', prediction=prediction, image_data=image_data, error=error)

if __name__ == '__main__':
    if model is None:
        raise RuntimeError('No trained model available. Train and save my_trained_network5.pth first.')
    app.run(host='0.0.0.0', port=5000, debug=True)
