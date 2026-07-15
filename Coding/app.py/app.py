from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and label encoder
model = joblib.load("hdi_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from the form
        life_expectancy = float(request.form["life_expectancy"])
        expected_schooling = float(request.form["expected_schooling"])
        mean_schooling = float(request.form["mean_schooling"])
        gni = float(request.form["gni"])

        # Prepare input for prediction
        input_data = np.array([[
            life_expectancy,
            expected_schooling,
            mean_schooling,
            gni
        ]])

        # Predict
        prediction = model.predict(input_data)
        result = label_encoder.inverse_transform(prediction)[0]

        return render_template(
            "index.html",
            prediction=f"Predicted Human Development: {result}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)