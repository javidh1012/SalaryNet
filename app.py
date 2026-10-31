import math
import os

import torch
from flask import Flask, jsonify, render_template, request

from main import SalaryNet


app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "salary_net.pth")
INR_PER_LAKH = 100_000

model = SalaryNet()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()


@app.get("/")
def home():
    return render_template("index.html", experience="", error=None, predicted_salary=None)


@app.post("/predict")
def predict():
    is_json = request.is_json
    data = request.get_json(silent=True) if is_json else request.form

    if not data or "experience" not in data:
        message = "Provide an 'experience' value."
        if is_json:
            return jsonify(error=message), 400
        return render_template(
            "index.html", experience="", error=message, predicted_salary=None
        ), 400

    try:
        experience = float(data["experience"])
    except (TypeError, ValueError):
        message = "'experience' must be a number."
        if is_json:
            return jsonify(error=message), 400
        return render_template(
            "index.html",
            experience=data["experience"],
            error=message,
            predicted_salary=None,
        ), 400

    if not math.isfinite(experience):
        message = "'experience' must be a finite number."
        if is_json:
            return jsonify(error=message), 400
        return render_template(
            "index.html",
            experience=data["experience"],
            error=message,
            predicted_salary=None,
        ), 400

    inputs = torch.tensor([[experience]], dtype=torch.float32)
    with torch.inference_mode():
        predicted_salary = model(inputs).item() * INR_PER_LAKH

    if is_json:
        return jsonify(predicted_salary=predicted_salary)
    return render_template(
        "index.html",
        experience=data["experience"],
        error=None,
        predicted_salary=predicted_salary,
    )


if __name__ == "__main__":
    app.run(host='0.0.0.0',port='8080',debug=False)
