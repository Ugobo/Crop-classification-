import io
import base64
from flask import Flask, request, render_template_string
from ultralytics import YOLO
from PIL import Image

app = Flask(__name__)
model = YOLO("best.pt")

PAGE = """
<!doctype html>
<title>Tomato Leaf Disease Classification</title>
<body style="font-family: Arial; max-width: 700px; margin: 40px auto; text-align: center;">
  <h2>Tomato Leaf Disease Classification</h2>
  <p>Upload a tomato leaf photo and the model will predict the disease.</p>
  <form method="post" enctype="multipart/form-data">
    <input type="file" name="image" accept="image/*" required>
    <button type="submit">Classify</button>
  </form>
  {% if preds %}
    <img src="data:image/jpeg;base64,{{ img_b64 }}" style="max-width:100%; margin-top:20px;">
    <h3>Prediction: {{ preds[0][0] }} ({{ preds[0][1] }}%)</h3>
    <p>Other possibilities:</p>
    {% for name, conf in preds[1:] %}
      <p>{{ name }}: {{ conf }}%</p>
    {% endfor %}
  {% endif %}
</body>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    preds = None
    img_b64 = None
    if request.method == "POST":
        file = request.files["image"]
        img = Image.open(file.stream).convert("RGB")
        result = model.predict(img, imgsz=224)[0]
        probs = result.probs
        top = probs.top5
        confs = probs.top5conf.tolist()
        preds = [(result.names[i].replace("_", " "), round(c * 100, 1))
                 for i, c in zip(top[:3], confs[:3])]
        img.thumbnail((500, 500))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()
    return render_template_string(PAGE, preds=preds, img_b64=img_b64)

if __name__ == "__main__":
    app.run(debug=True)