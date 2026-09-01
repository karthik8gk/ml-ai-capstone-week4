# Week 4 — AI Project Deployment & Capstone

## Student Score Prediction API

An end-to-end machine-learning application that covers data preparation, model training, evaluation, serialization, and deployment through a Flask prediction API.

### Workflow
1. Load the student-performance dataset.
2. Separate features and target.
3. Handle numerical and categorical preprocessing inside a pipeline.
4. Train a Random Forest Regressor.
5. Evaluate the model using MAE, RMSE and R².
6. Serialize the complete pipeline with Joblib.
7. Load the saved model from a Flask application.
8. Expose `/predict` and `/health` API endpoints.
9. Provide a simple browser interface for predictions.

### Run locally

```bash
pip install -r requirements.txt
python src/train_model.py
python src/app.py
```

Open `http://127.0.0.1:5000/`.

### API example

POST `/predict` with JSON:

```json
{
  "Age": 21,
  "Attendance": 85,
  "Study_Hours": 5,
  "Previous_Score": 78,
  "Department": "CSE"
}
```

Response:

```json
{
  "predicted_final_score": 84.23
}
```

### Project structure

```text
ml-ai-capstone-week4/
├── data/
├── model/
├── src/
├── templates/
├── tests/
├── Dockerfile
├── Procfile
├── README.md
└── requirements.txt
```

### Deployment note
The application is packaged for deployment with Flask and also includes a Dockerfile and Procfile. The saved Joblib model is loaded when the API starts.
