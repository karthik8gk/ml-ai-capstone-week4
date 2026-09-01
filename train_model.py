from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE=Path(__file__).resolve().parents[1]
df=pd.read_csv(BASE/"data/student_performance.csv")
X=df[["Age","Attendance","Study_Hours","Previous_Score","Department"]]
y=df["Final_Score"]

num=["Age","Attendance","Study_Hours","Previous_Score"]
cat=["Department"]
pre=ColumnTransformer([
("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),num),
("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)
])
model=Pipeline([
("preprocessor",pre),
("regressor",RandomForestRegressor(n_estimators=150,max_depth=8,random_state=42))
])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42)
model.fit(Xtr,ytr)
pred=model.predict(Xte)
metrics={
"MAE":mean_absolute_error(yte,pred),
"RMSE":mean_squared_error(yte,pred)**0.5,
"R2":r2_score(yte,pred)
}
pd.DataFrame([metrics]).to_csv(BASE/"model/evaluation_metrics.csv",index=False)
joblib.dump(model,BASE/"model/student_score_model.joblib")
print("Model saved.")
print(metrics)
