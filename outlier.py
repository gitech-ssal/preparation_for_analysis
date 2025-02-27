# ----------------------------------------------------------------------------------------------------------
# 라이브러리 목록

# 기본 라이브러리 
import numpy as np
import pandas as pd
# ----------------------------------------------------------------------------------------------------------





# ----------------------------------------------------------------------------------------------------------
class OutlierHandler:
    def __init__(self, df):
        """
        데이터프레임을 받아 초기화합니다.
        """
        self.df = df
        self.outlier_indices = {}  # 각 컬럼별 이상치 인덱스를 저장

    def calculate_bounds(self, column):
        """
        특정 컬럼의 이상치 하한값과 상한값을 반환합니다.
        """
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return lower_bound, upper_bound

    def find_outliers(self, column):
        """
        특정 컬럼에서 이상치 인덱스를 반환하고 저장합니다.
        """
        lower_bound, upper_bound = self.calculate_bounds(column)
        outliers = self.df.index[(self.df[column] < lower_bound) | (self.df[column] > upper_bound)]
        self.outlier_indices[column] = list(outliers)
        return list(outliers)

    def find_outliers_multiple(self, columns):
        """
        여러 컬럼에서 이상치 인덱스를 탐색하고 저장합니다.
        """
        for column in columns:
            self.find_outliers(column)
        return self.outlier_indices

    def remove_outliers(self, columns=None):
        """
        지정된 컬럼의 이상치를 제거한 데이터프레임 반환.
        """
        if columns is None:
            columns = list(self.outlier_indices.keys())

        outlier_idx = set()
        for column in columns:
            if column not in self.outlier_indices:
                self.find_outliers(column)
            outlier_idx.update(self.outlier_indices[column])

        cleaned_df = self.df.drop(index=outlier_idx)
        return cleaned_df

    def get_outlier_info(self, column):
        """
        특정 컬럼의 이상치 정보를 반환합니다.
        """
        if column not in self.outlier_indices:
            self.find_outliers(column)
        lower_bound, upper_bound = self.calculate_bounds(column)
        return {"lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outliers": self.outlier_indices[column],}
# ----------------------------------------------------------------------------------------------------------