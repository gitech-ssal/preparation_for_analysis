# ----------------------------------------------------------------------------------------------------
# 라이브러리 목록 

# 기본 라이브러리
import re
import pandas as pd
# ----------------------------------------------------------------------------------------------------




# ----------------------------------------------------------------------------------------------------
class DataFrameEncoder:
    """
    DataFrame의 열을 기반으로 원-핫 인코딩과 라벨 인코딩 딕셔너리를 생성 및 변환하는 클래스.
    """
    def __init__(self, df=None, columns=None, ascending_order=True, sort_by_number=False):
        if df is None or columns is None:
            raise ValueError("Both 'df' and 'columns' must be provided.")

        if isinstance(columns, str):
            columns = [columns]  # 문자열인 경우 리스트로 변환

        if isinstance(ascending_order, bool):
            ascending_order = [ascending_order] * len(columns)
        if isinstance(sort_by_number, bool):
            sort_by_number = [sort_by_number] * len(columns)

        if len(columns) != len(ascending_order):
            raise ValueError("Length of 'ascending_order' must match the number of columns.")
        if len(columns) != len(sort_by_number):
            raise ValueError("Length of 'sort_by_number' must match the number of columns.")

        missing_columns = [col for col in columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Columns {missing_columns} do not exist in the DataFrame.")

        self.df = df
        self.columns = columns
        self.ascending_order = ascending_order
        self.sort_by_number = sort_by_number
        self.unique_values = {col: self._get_unique_values(col) for col in self.columns}

    def _extract_number(self, value):
        """
        문자열에서 숫자 부분을 추출하여 반환.
        :param value: 입력 값
        :return: 숫자 부분 (없으면 큰 값을 반환)
        """
        value = str(value)  # 문자열로 변환
        match = re.search(r'(\d+)', value)
        return int(match.group(0)) if match else float('inf')

    def _get_unique_values(self, column):
        """
        특정 열의 고유 값을 정렬하여 반환.
        :param column: 정렬할 열 이름
        :return: 정렬된 고유 값 리스트
        """
        col_index = self.columns.index(column)
        ascending = self.ascending_order[col_index]
        sort_by_number = self.sort_by_number[col_index]

        unique_values = self.df[column].drop_duplicates()

        if unique_values.empty:
            return pd.Series(dtype='object')  # 빈 Series 반환

        if sort_by_number:
            # 숫자 기준으로 정렬
            sort_keys = unique_values.apply(self._extract_number)
            sorted_indices = sort_keys.argsort(kind='mergesort')
            if not ascending:
                sorted_indices = sorted_indices[::-1]
            unique_values = unique_values.iloc[sorted_indices]
        else:
            # 기본 알파벳 순서로 정렬
            unique_values = unique_values.sort_values(ascending=ascending)

        return unique_values.reset_index(drop=True)

    def get_onehot_dict(self, column):
        """
        특정 열의 원-핫 인코딩 딕셔너리를 생성.
        :param column: 열 이름
        :return: 원-핫 인코딩 딕셔너리
        """
        values = self.unique_values[column]
        return {value: [1 if i == idx else 0 for i in range(len(values))] for idx, value in enumerate(values)}

    def get_label_dict(self, column):
        """
        특정 열의 라벨 인코딩 딕셔너리를 생성.
        :param column: 열 이름
        :return: 라벨 인코딩 딕셔너리
        """
        values = self.unique_values[column]
        return {value: idx for idx, value in enumerate(values)}

    def fit_transform(self, encoding_type="onehot", custom_dicts=None):
        """
        데이터를 지정된 인코딩 방식으로 변환.
        :param encoding_type: 'onehot' 또는 'label'
        :param custom_dicts: 사용자 정의 딕셔너리 (옵션)
        :return: 변환된 데이터프레임
        """
        if encoding_type not in ["onehot", "label"]:
            raise ValueError("encoding_type must be 'onehot' or 'label'.")

        transformed_data = pd.DataFrame()

        for column in self.columns:
            # 사용자 정의 딕셔너리 확인
            if custom_dicts and column in custom_dicts:
                encoder_dict = custom_dicts[column]
            else:
                encoder_dict = (self.get_onehot_dict(column) if encoding_type == "onehot" else self.get_label_dict(column))

            # 데이터 변환
            encoded_data = self.df[column].map(encoder_dict)

            if encoding_type == "onehot":
                onehot_df = pd.DataFrame(encoded_data.tolist(), columns=list(encoder_dict.keys()))
                transformed_data = pd.concat([transformed_data, onehot_df], axis=1)
            else:
                transformed_data[column] = encoded_data

        return transformed_data
# ----------------------------------------------------------------------------------------------------