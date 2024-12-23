import plotly.graph_objects as go
from analyze_team import load_data
import os
import pandas as pd
import numpy as np

def create_bar_chart(x, y, title, x_title, y_title):
    """
    막대 그래프 생성 함수
    """
    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    return fig

def create_detail_bar_chart(x, y, title, x_title, y_title):
    """
    세부 기준별 막대 그래프 생성 함수 (수정)
    """
    fig = go.Figure(data=[go.Bar(x=x, y=y)])
    fig.update_layout(title=title, xaxis_title=x_title, yaxis_title=y_title)
    return fig

def visualize_data(season):
    """
    데이터 시각화 함수
    """
    stat_array, dimension_info = load_data(season)

    # 선수 ID 목록 가져오기
    player_ids = dimension_info['dimensions']['player']

    # 모든 분석 지표에 대해 시각화
    for analysis_index, analysis_type in enumerate(dimension_info['dimensions']['analysis']):
        # 선수별 시각화
        for player_index, player_id in enumerate(player_ids):
            # 해당 분석 기준, 해당 선수에 대한 데이터 추출
            stat_data = stat_array[player_index, analysis_index, :, :]

            # 시각화
            if any(dimension_info['dimensions']['detail'][analysis_index]):  # 세부 기준이 있는 경우
                # 세부 기준별 시각화 (수정)
                detail_labels = dimension_info['dimensions']['detail'][analysis_index]
                
                # NaN 값을 포함하는 행 제거
                mask = ~np.isnan(stat_data).any(axis=1)
                stat_data_filtered = stat_data[mask]
                detail_labels_filtered = [label for i, label in enumerate(detail_labels) if mask[i]]

                y_data = stat_data_filtered[:, 0] # 예: 첫 번째 통계 필드 사용

                fig = create_detail_bar_chart(detail_labels_filtered, y_data, f"{season} 시즌 {player_id} {analysis_type}", analysis_type, dimension_info['stat_fields'][0])
            else:
                # 전체 기준 시각화 (세부 기준 없음)
                x = [analysis_type]
                y = stat_data.mean(axis=0)  # stat 차원에 대해 평균 계산
                fig = create_bar_chart(x, y, f"{season} 시즌 {player_id} {analysis_type}", analysis_type, "평균")

            # HTML 파일로 저장
            output_dir = os.path.join(os.path.dirname(__file__), "output", analysis_type) # 분석 유형별로 폴더 생성
            os.makedirs(output_dir, exist_ok=True)
            fig.write_html(os.path.join(output_dir, f"{season}_{player_id}_{analysis_type}.html"))

if __name__ == "__main__":
    season = "2024"
    visualize_data(season)