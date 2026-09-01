from src.app import app
def test_health():
    c=app.test_client()
    r=c.get('/health')
    assert r.status_code==200
    assert r.json['status']=='ok'
def test_predict():
    c=app.test_client()
    r=c.post('/predict',json={'Age':21,'Attendance':85,'Study_Hours':5,'Previous_Score':78,'Department':'CSE'})
    assert r.status_code==200
    assert 'predicted_final_score' in r.json
