import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI
from pydantic import BaseModel, Field
import os

# FastAPI 앱 인스턴스 생성
app = FastAPI()

# 1. Spring에서 받을 원본 데이터의 형식을 정의합니다.
# Pydantic의 Field(alias=...)를 사용하여 카멜 케이스 변수명을 매핑합니다.
class PredictionRequest(BaseModel):
    user_id: int = Field(..., alias="userId")
    name: str
    age: float
    gender: int
    education: str
    home_ownership: str = Field(..., alias="homeOwnership")

    # 금융 정보
    loan_id: int = Field(..., alias="loanId")
    income: float
    emp_exp: int = Field(..., alias="empExp")
    amount: float
    intent: str
    int_rate: float = Field(..., alias="intRate")
    loan_percent_income: float = Field(..., alias="loanPercentIncome")
    cred_hist_length: float = Field(..., alias="credHistLength")
    credit_score: int = Field(..., alias="creditScore")
    previous_loan_defaults: int = Field(..., alias="previousLoanDefaults")
    # loan_status: int = Field(..., alias="loanStatus")

    class Config:
        populate_by_name = True

# 2. 모델, 인코더, Explainer를 전역 변수로 선언
model = None
encoder = None
explainer = None
feature_names = None

# FastAPI 서버가 시작될 때 실행되는 이벤트 핸들러
@app.on_event("startup")
async def load_resources():
    """
    서버 시작 시 모델, 인코더, Explainer를 로드합니다.
    """
    global model, encoder, explainer, feature_names
    
    try:
        # 파일 경로 설정
        model_path = os.path.join("model", "xgb_model_log.pkl")
        encoder_path = os.path.join("model", "ohe_encoder.pkl")
        train_data_path = os.path.join("data", "model_data_encoded.csv")
        
        # 2.1. 학습된 모델 및 OneHotEncoder 객체 로드
        model = joblib.load(model_path)
        encoder = joblib.load(encoder_path)
        
        # 2.2. SHAP Explainer를 위한 데이터셋 로드
        df_train_encoded = pd.read_csv(train_data_path) 
        
        # 2.3. 모델 학습 시 사용한 특성 순서 정의
        feature_names = df_train_encoded.columns.tolist()
        
        # 2.4. SHAP Explainer 객체 생성
        explainer = shap.TreeExplainer(model)
        explainer.feature_names = feature_names
        
        print("필요한 모든 객체 로드 완료.")
        
    except FileNotFoundError as e:
        print(f"오류: {e}. 필요한 파일이 존재하지 않습니다. 서버를 시작할 수 없습니다.")
        raise RuntimeError("서버 시작 실패: 필수 파일 누락.")
    except Exception as e:
        print(f"객체 로드 중 예상치 못한 오류 발생: {e}")
        raise

# 3. API 엔드포인트 정의
@app.post("/predict_and_explain")
def predict_with_shap(request_data: PredictionRequest):
    """
    Spring에서 원본 데이터를 받아 전처리, 예측 및 SHAP 값을 반환합니다.
    """

    print("Received JSON from Spring Boot:")
    print(request_data.model_dump_json(indent=2))

    # 3.1. Spring에서 받은 데이터를 DataFrame으로 변환
    input_df = pd.DataFrame([request_data.model_dump(by_alias=False)])
    
    # 3.2. 로그 변환 수행
    input_df['income'] = np.log1p(input_df['income'])

    # 3.3. 원핫 인코딩 수행
    categorical_features = ['education', 'home_ownership', 'intent']
    encoded_features = encoder.transform(input_df[categorical_features])
    
    # 3.4. 인코딩된 데이터를 DataFrame으로 변환
    encoded_df = pd.DataFrame(
        encoded_features,
        columns=encoder.get_feature_names_out(categorical_features)
    )

    # 3.5. 최종 입력 데이터 구성
    # 원본 데이터의 수치형 특성과 인코딩된 범주형 특성을 병합
    numerical_features = ['age', 'gender', 'emp_exp', 'amount', 'int_rate', 'loan_percent_income', 'cred_hist_length', 'credit_score', 'previous_loan_defaults']
    final_df = pd.concat([input_df[['income']], input_df[numerical_features], encoded_df], axis=1)

    # 3.6. 특성 순서 맞추기 (매우 중요)
    final_input_df = final_df.reindex(columns=feature_names, fill_value=0)
    input_data_np = final_input_df.values

    # 3.7. 모델 예측 및 SHAP 값 계산
    prediction = model.predict(input_data_np)[0]
    shap_values = explainer.shap_values(input_data_np)[0]

    # 3.8. 응답 데이터 구성 및 반환
    response_data = {
        "prediction": float(prediction),
        "shap_values": shap_values.tolist()
    }
    
    return response_data
