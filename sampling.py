import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

class ConditionalKDESampler:
    def __init__(self, df, model_col, warranty_col, condition_col, price_col):
        """
        "Model", "Warranty", "Color" 조건에 맞는 확률 밀도 추정 및 샘플링 클래스

        :param df: 입력 데이터프레임
        :param model_col: 모델이 되는 컬럼명 (예: "Model")
        :param warranty_col: 보증기간이 되는 컬럼명 (예: "Warranty")
        :param condition_col: 차량상태 컬럼 (예: "Condition")
        :param price_col: 확률 밀도를 구할 대상 값 컬럼 (예: "Price")
        """
        self.model_col = model_col
        self.warranty_col = warranty_col
        self.condition_col = condition_col
        self.price_col = price_col
        self.kde_dict = self._fit_kdes(df)
    
    def _fit_kdes(self, df):
        """
        조건 ("Model", "Warranty", "Condition")에 맞는 KDE 모델을 생성하여 저장
        """
        kde_dict = {}
        grouped = df.groupby([self.model_col, self.warranty_col, self.condition_col])
        
        for (model, warranty, condition), group in grouped:
            values = group[self.price_col].values
            if len(values) > 1:  # 데이터가 1개 이상 있어야 KDE 가능
                kde_dict[(model, warranty, condition)] = gaussian_kde(values)
            else:
                kde_dict[(model, warranty, condition)] = None  # 데이터 부족 시 KDE 없음
        
        return kde_dict

    def sample(self, model, warranty, condition, n_samples=1):
        """
        특정 "Model", "Warranty", "Color" 조건에 대한 KDE에서 샘플링하여 값 반환
        :param model: 모델 값
        :param warranty: 보증 기간 값
        :param condition: 차량상태 값 
        :param n_samples: 샘플링 횟수
        :return: 샘플링된 값 (또는 None)
        """
        kde = self.kde_dict.get((model, warranty, condition))
        if kde is not None:
            sampled_values = kde.resample(n_samples)[0]  # 샘플링된 값을 배열로 반환
            sampled_values = np.round(sampled_values, 2)  # 소수점 2자리로 반올림
            return sampled_values  # 배열 그대로 반환
        else:
            return None  # 해당 조건에 대한 KDE 없음

