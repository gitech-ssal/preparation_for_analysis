# ----------------------------------------------------------------------------------------------------
# 라이브러리 목록 

# 기본 라이브러리 
import json
import pandas as pd
# ----------------------------------------------------------------------------------------------------




# ----------------------------------------------------------------------------------------------------------
class DataVisualizer:
    def __init__(self, line="=", length=50, start="#"):
        """
        초기화 메서드
        :param line: 구분선 문자
        :param length: 구분선 길이
        :param start: 출력 시작 문자
        """
        self.line = line
        self.length = length
        self.start = start

    def _print_separator(self):
        """
        구분선을 출력
        """
        print(self.line * self.length)

    def _space(self, num):
        """
        공백을 출력
        """
        print("\n"*num)

    def show_data(self, title=None, data=None):
        """
        일반 데이터를 출력
        """
        self._print_separator()
        print(f"\n{self.start} Title: {title}\n")
        self._print_separator()
        print(f"\n{self.start} Data:\n")
        print(data)
        print()
        self._print_separator()
        self._space(num=3)

    def show_dict(self, title=None, dictionary=None):
        """
        딕셔너리를 깔끔하게 출력
        """
        self._print_separator()
        print(f"\n{self.start} Title: {title}\n")
        self._print_separator()
        print(f"\n{self.start} Inside of Dictionay:\n")
        if dictionary:
            max_key_len = max(len(str(key)) for key in dictionary.keys())
            for key, value in dictionary.items():
                print(f"{key:<{max_key_len}} : {value}")
        else:
            print("No dictionary provided.")
        print()
        self._print_separator()
        self._space(num=3)

    def show_df_info(self, title=None, df=None):
        """
        DataFrame 정보를 출력
        """
        self._print_separator()
        print(f"\n{self.start} Title: {title}\n")
        self._print_separator()
        print(f"\n{self.start} DataFrame Information:\n")
        if df is not None:
            df.info()
        else:
            print("No DataFrame provided.")
        print()
        self._print_separator()
        self._space(num=3)

    def show_json(self, title=None, json_data=None):
        """JSON 데이터를 보기 좋게 출력"""
        self._print_separator()
        print(f"\n{self.start} Title: {title}\n")
        print(f"\n{self.start} JSON Data:\n")
        self._print_separator()
        if json_data:
            print(json.dumps(json_data, indent=4))
        else:
            print("No JSON Data provided.")
        print()
        self._print_separator()
        self._space(num=3)
# ----------------------------------------------------------------------------------------------------------