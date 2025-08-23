# from fastapi import FastAPI
# # fastapi는 8000번 포트 사용

# # FastAPI 앱 인스턴스 생성
# app = FastAPI()

# # 루트 경로('/')에 대한 GET 요청 처리
# @app.get("/")
# def read_root():
#     return {"message": "Hello World"}

import joblib
import numpy as np
import shap
from fastapi import FastAPI
from pydantic import BaseModel

# FastAPI 앱 인스턴스 생성
app = FastAPI()

# 1. Spring에서 받을 JSON 데이터의 형식을 정의합니다.
# Pydantic을 사용하여 데이터 유효성 검사를 자동으로 처리할 수 있습니다.
class PredictionRequest(BaseModel):
    # Spring에서 보낼 데이터의 특성 이름과 타입을 정의합니다.
    # 여기서는 예시로 'feature1', 'feature2', 'feature3'를 사용합니다.
    feature1: float
    feature2: float
    feature3: float

# 2. 모델과 SHAP Explainer를 로드합니다.
# 실제 프로젝트에서는 모델을 미리 학습시켜 my_model.pkl 파일로 저장해야 합니다.
try:
    # 예시 모델 로드 (실제 모델 파일 경로로 변경 필요)
    model = joblib.load("model/xgb_model_log.pkl")
    
    # SHAP Explainer 객체 생성 (모델과 학습 데이터로 생성)
    # SHAP explainer를 생성하기 위한 예시 데이터 (모델 학습에 사용된 데이터와 유사해야 함)
    # 실제로는 학습 데이터셋을 로드하여 사용해야 합니다.
    dummy_data = np.array([[10, 20, 30], [15, 25, 35], [5, 10, 15]])
    explainer = shap.Explainer(model, dummy_data)
except FileNotFoundError:
    # 모델 파일이 없을 경우 더미 모델과 explainer를 생성합니다.
    print("모델 파일을 찾을 수 없습니다. 예시 더미 모델을 사용합니다.")
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    # 더미 데이터로 모델 학습
    X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y_train = np.array([10, 20, 30])
    model.fit(X_train, y_train)
    dummy_data = X_train
    explainer = shap.Explainer(model, dummy_data)
    
# 3. 새로운 API 엔드포인트를 정의합니다.
@app.post("/predict_and_explain")
def predict_with_shap(request_data: PredictionRequest):
    """
    Spring에서 JSON 데이터를 받아 예측을 수행하고 SHAP 값을 반환합니다.
    """
    # 4. JSON 데이터를 NumPy 배열로 변환합니다.
    # Pydantic 모델의 데이터를 딕셔너리로 변환 후, 값을 추출하여 2차원 배열로 만듭니다.
    input_data = np.array([[
        request_data.feature1,
        request_data.feature2,
        request_data.feature3
    ]])

    # 5. 모델로 예측을 수행합니다.
    prediction = model.predict(input_data)[0]

    # 6. SHAP 값을 계산합니다.
    shap_values = explainer.shap_values(input_data)[0]

    # 7. 응답을 위한 JSON 데이터를 구성합니다.
    # 예측 결과와 SHAP 값을 함께 반환합니다.
    # SHAP 값은 numpy.ndarray이므로 list로 변환해야 JSON으로 직렬화할 수 있습니다.
    response_data = {
        "prediction": float(prediction),
        "shap_values": shap_values.tolist()
    }
    
    return response_data