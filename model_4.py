# ----------------------------------------------------------------------------------------------------
# 라이브러리 목록 

# 기본 라이브러리 
import os
import numpy as np
import pandas as pd

# sklearn 라이브러리 
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

# xgboost 라이브러리 
from xgboost import XGBRegressor

# 파라미터 최적화 라이브러리 
import optuna

# 파일 라이브러리 
import joblib

# 경고 설정 라이브러리 
import warnings

# 경고 설정 
warnings.filterwarnings('ignore')
# ----------------------------------------------------------------------------------------------------





# ----------------------------------------------------------------------------------------------------
class XGBPricePredictor:
    def __init__(self,
                 model_dir = "C:\\Users\\ssalt\\Documents\\ev_price_predict_project\\data\\train\\B_models\\d_model_4",
                 filename = "best_model.joblib",
                 n_splits = 5, 
                 random_state=1234):
        
        # 모델 디렉토리 및 주소
        self.model_dir = model_dir
        self.model_path = os.path.join(self.model_dir, filename)
        os.makedirs(self.model_dir, exist_ok=True) # 모델 디렉토리 검사 

        # optimize_hyperparameters 설정 
        self.n_splits = n_splits
        self.random_state = random_state

        # 모델 변수 
        self.best_model = None
        self.best_params = None

    def objective(self, trial, x_train, y_train):

        # 파라미터 
        param = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.01),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
            "objective": trial.suggest_categorical("objective", ["reg:squarederror"]),
            "eval_metric": trial.suggest_categorical("eval_metric", ["rmse"]),
            "n_jobs": trial.suggest_int("n_jobs", 1, 8),
            "tree_method": trial.suggest_categorical("tree_method", ["auto", "gpu_hist"]),
        }

        # 모델 선언 
        model = XGBRegressor(random_state=self.random_state, **param)

        # 모델 fold 
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        # 모델 평가 점수 
        rmse_scores = []

        for train_idx, valid_idx in cv.split(x_train):
            # train을 다시 X_train,Y_train,X_valid,Y_valid 분리
            X_train, X_valid = x_train.iloc[train_idx], x_train.iloc[valid_idx]
            Y_train, Y_valid = y_train.iloc[train_idx], y_train.iloc[valid_idx]

            # 모델 학습
            model.fit(X_train, Y_train, 
                      eval_set=[(X_valid, Y_valid)],
                      verbose=0)
            
            # 모델 예측 
            preds = model.predict(X_valid)

            # 모델 평가 
            rmse = np.sqrt(mean_squared_error(Y_valid, preds))
            rmse_scores.append(rmse)

        return np.mean(rmse_scores)

    def optimize_hyperparameters(self, x_train, y_train, n_trials=50):

        # 모델의 파라미터 변수 초기화 
        self.best_params = None

        # optuna 선언 
        study = optuna.create_study(direction="minimize")

        # 최적화 탐구 
        study.optimize(lambda trial: self.objective(trial, x_train, y_train),
                        n_trials=n_trials,
                        show_progress_bar=True)

        # 최적화 결과 파라미터 저장 
        self.best_params = study.best_params

        return self.best_params

    def train(self, x_train, y_train):

        # 에러 방지 
        if not self.best_params:
            raise ValueError("먼저 `optimize_hyperparameters`를 호출하세요.")
        
        # best_model 선언 
        self.best_model = XGBRegressor(random_state=self.random_state, **self.best_params)

        # best_model 훈련 
        self.best_model.fit(x_train, y_train, eval_set=[(x_train, y_train)], verbose=1)

    def evaluate(self, x_test, y_test):

        # 에러 방지 
        if not self.best_model:
            raise ValueError("모델이 훈련되지 않았습니다. `train`을 먼저 호출하세요.")
        
        # 예측 실시 
        preds = self.best_model.predict(x_test)

        # 예측 평가 
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        return {"RMSE": rmse, "R²": r2}

    def save_model(self):

        # 에러 방지 
        if not self.best_model:
            raise ValueError("모델이 훈련되지 않았습니다. `train`을 먼저 호출하세요.")
        
        # 모델 저장 
        joblib.dump(self.best_model, self.model_path)

        # 저장 완료 
        print(f"Model saved to {self.model_path}.")

    def load_model(self):

        # 에러 방지 
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"{self.model_path} 경로에 모델이 존재하지 않습니다.")
        
        # 모델 로드 
        self.best_model = joblib.load(self.model_path)

        # 로드된 모델에서 파라미터 가져오기
        self.best_params = self.best_model.get_params()

        # 로드 완료 
        print(f"Model loaded from {self.model_path}.")
# ----------------------------------------------------------------------------------------------------
