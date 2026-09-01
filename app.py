from pathlib import Path
import joblib
from flask import Flask, request, jsonify, render_template

BASE=Path(__file__).resolve().parents[1]
MODEL_PATH=BASE/"model/student_score_model.joblib"
model=joblib.load(MODEL_PATH)

app=Flask(__name__)

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify({"status":"ok","service":"student-score-prediction-api"})

@app.post("/predict")
def predict():
    data=request.get_json(silent=True)
    required=["Age","Attendance","Study_Hours","Previous_Score","Department"]
    if not data:
        return jsonify({"error":"Request body must be valid JSON."}),400
    missing=[x for x in required if x not in data]
    if missing:
        return jsonify({"error":"Missing fields","fields":missing}),400
    try:
        row={k:[data[k]] for k in required}
        prediction=float(model.predict(__import__("pandas").DataFrame(row))[0])
        return jsonify({"predicted_final_score":round(prediction,2)})
    except Exception as exc:
        return jsonify({"error":str(exc)}),400

if __name__=="__main__":
    app.run(host="0.0.0.0",port=5000,debug=False)
