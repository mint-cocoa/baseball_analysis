import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

STAT_FIELDS = [
    'games', 'plate_appearances', 'at_bats', 'hits', 'home_runs', 'runs', 'rbis',
    'walks', 'strikeouts', 'stolen_bases', 'singles', 'doubles', 'triples',
    'intentional_walks', 'hit_by_pitch', 'sacrifice_flies', 'sacrifice_bunts',
    'gidp', 'caught_stealing', 'babip', 'avg', 'obp', 'slg', 'ops', 'woba',
    'war', 'bb_pct', 'k_pct', 'bb_k_ratio', 'iso', 'ab_hr_ratio', 'rc',
    'rc27', 'wrc', 'spd', 'wsb', 'wraa'
]

CORE_STAT_INDICES = {
    'war': STAT_FIELDS.index('war'),
    'ops': STAT_FIELDS.index('ops'),
    'woba': STAT_FIELDS.index('woba'),
    'bb_k_ratio': STAT_FIELDS.index('bb_k_ratio'),
    'rc27': STAT_FIELDS.index('rc27'),
    'spd': STAT_FIELDS.index('spd')
}

CORE_STATS = ['war', 'ops', 'woba', 'bb_k_ratio', 'rc27', 'spd']
CORE_STAT_DESCRIPTIONS = {
    'war': '승리 기여도',
    'ops': '출루율+장타율',
    'woba': '가중 출루율',
    'bb_k_ratio': '볼넷/삼진 비율',
    'rc27': '27아웃당 득점 생산력',
    'spd': '주루 능력'
}

PLAYER_NAMES = {
    '738': '강민호',  # 136경기
    '1754': '김지찬',  # 135경기
    '517': '구자욱',  # 129경기
    '2070': '김영웅',  # 126경기
    '1436': '이성규',  # 122경기
    '478': '박병호',  # 120경기
    '1990': '김헌곤',  # 117경기
    '704': '이재현',  # 109경기
    '2073': '류지혁',  # 100경기
    '1499': '이병헌',  # 95경기
    '2178': '안주형',  # 82경기
    '2005': '김현준',  # 79경기
    '86': '맥키넌',  # 72경기
    '2121': '윤정빈',  # 69경기
    '1508': '전병우',  # 58경기
    '2262': '김재상',  # 35경기
    '2267': '김재혁',  # 35경기
    '1285': '김성윤',  # 32경기
    '1321': '김동진',  # 30경기
    '871': '디아즈',  # 29경기
    '2250': '김호진',  # 26경기
    '2143': '강한울',  # 18경기
    '1687': '양도근'   # 16경기
}

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def load_data(season):
    """데이터 로드"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    stat_array = np.load(os.path.join(script_dir, "processed", f"stat_array_{season}.npy"))
    dimension_info = np.load(os.path.join(script_dir, "processed", f"dimension_info_{season}.npy"), allow_pickle=True).item()
    return stat_array, dimension_info

def display_player_data(stat_array, dimension_info, player_id):
    """특정 선수의 데이터를 분석 유형 및 세부 기준별로 출력"""
    try:
        player_index = dimension_info['dimensions']['player'].index(player_id)
    except ValueError:
        print(f"오류: 선수 ID '{player_id}'를 찾을 수 없습니다.")
        return

    player_data = stat_array[player_index]
    analysis_names = dimension_info['dimensions']['analysis']
    stat_names = dimension_info['stat_fields']

    print(f"\n{player_id} 선수의 데이터:")

    for analysis_index, analysis_name in enumerate(analysis_names):
        print(f"\n## {analysis_name}:")
        detail_names = dimension_info['dimensions']['detail'][analysis_index]

        for detail_index, detail_name in enumerate(detail_names):
            if detail_name is None:
                continue

            print(f"\n### {detail_name}:")
            detail_data = player_data[analysis_index, detail_index]

            for stat_index, stat_name in enumerate(stat_names):
                print(f"- {stat_name}: {detail_data[stat_index]}")

def analyze_core_stats(stat_array, dimension_info):
    """핵심 성과 지표 분석"""
    players = dimension_info['dimensions']['player']
    total_index = dimension_info['dimensions']['analysis'].index('total')
    
    # 전체 성적 데이터 추출
    total_stats = {}
    for player_id in players:
        player_index = dimension_info['dimensions']['player'].index(player_id)
        player_data = stat_array[player_index, total_index, 0]
        
        player_stats = {}
        for stat_name in CORE_STATS:
            stat_index = STAT_FIELDS.index(stat_name)
            player_stats[stat_name] = player_data[stat_index]
        total_stats[player_id] = player_stats
    
    # DataFrame으로 변환
    df = pd.DataFrame.from_dict(total_stats, orient='index')
    
    # 선수 이름 추가
    df['player_name'] = df.index.map(lambda x: PLAYER_NAMES.get(x, f"선수 {x}"))
    
    # 각 지표별 상위 5명 출력
    print("\n=== 핵심 지표별 상위 5명 ===")
    for stat in CORE_STATS:
        print(f"\n## {CORE_STAT_DESCRIPTIONS[stat]} ({stat})")
        top_players = df.nlargest(5, stat)
        print(f"{top_players[['player_name', stat]].to_string()}")
    
    return df

def analyze_monthly_trend(stat_array, dimension_info):
    """월별 성적 추이 분석"""
    players = dimension_info['dimensions']['player']
    month_index = dimension_info['dimensions']['analysis'].index('month')
    
    monthly_stats = {}
    for player_id in players:
        player_index = dimension_info['dimensions']['player'].index(player_id)
        player_monthly = {}
        
        for month_idx, month in enumerate(dimension_info['dimensions']['detail'][month_index]):
            if month is None:
                continue
            
            month_data = stat_array[player_index, month_index, month_idx]
            stats = {}
            for stat_name in CORE_STATS:
                stat_index = STAT_FIELDS.index(stat_name)
                stats[stat_name] = month_data[stat_index]
            player_monthly[month] = stats
        
        monthly_stats[player_id] = player_monthly
    
    return pd.DataFrame.from_dict({(i,j): monthly_stats[i][j] 
                                 for i in monthly_stats.keys() 
                                 for j in monthly_stats[i].keys()},
                                 orient='index')

def analyze_situational_stats(stat_array, dimension_info):
    """상황별 성적 분석"""
    players = dimension_info['dimensions']['player']
    base_index = dimension_info['dimensions']['analysis'].index('base')
    
    situation_stats = {}
    for player_id in players:
        player_index = dimension_info['dimensions']['player'].index(player_id)
        player_situations = {}
        
        for sit_idx, situation in enumerate(dimension_info['dimensions']['detail'][base_index]):
            if situation is None:
                continue
            
            sit_data = stat_array[player_index, base_index, sit_idx]
            stats = {}
            for stat_name in ['avg', 'ops', 'woba']:  # 주요 타격 지표만 선택
                stat_index = STAT_FIELDS.index(stat_name)
                stats[stat_name] = sit_data[stat_index]
            player_situations[situation] = stats
        
        situation_stats[player_id] = player_situations
    
    return pd.DataFrame.from_dict({(i,j): situation_stats[i][j] 
                                 for i in situation_stats.keys() 
                                 for j in situation_stats[i].keys()},
                                 orient='index')

def create_player_report(player_id, stat_array, dimension_info):
    """개별 선수 상세 보고서 생성"""
    try:
        player_index = dimension_info['dimensions']['player'].index(player_id)
    except ValueError:
        print(f"오류: 선수 ID '{player_id}'를 찾을 수 없습니다.")
        return
    
    total_index = dimension_info['dimensions']['analysis'].index('total')
    total_data = stat_array[player_index, total_index, 0]
    
    player_name = PLAYER_NAMES.get(player_id, f"선수 {player_id}")
    report = f"# {player_name} 선수 상세 분석 보고서\n\n"
    
    # 기본 성적
    report += "## 1. 기본 성적\n"
    for stat in ['games', 'plate_appearances', 'avg', 'obp', 'slg', 'ops', 'war']:
        stat_index = STAT_FIELDS.index(stat)
        report += f"- {stat}: {total_data[stat_index]:.3f}\n"
    
    # 타격 세부 지표
    report += "\n## 2. 타격 세부 지표\n"
    for stat in ['hits', 'doubles', 'triples', 'home_runs', 'walks', 'strikeouts']:
        stat_index = STAT_FIELDS.index(stat)
        report += f"- {stat}: {total_data[stat_index]:.0f}\n"
    
    # 득점 생산력
    report += "\n## 3. 득점 생산력\n"
    for stat in ['runs', 'rbis', 'rc', 'rc27', 'wrc']:
        stat_index = STAT_FIELDS.index(stat)
        report += f"- {stat}: {total_data[stat_index]:.2f}\n"
    
    return report

def main():
    logger = setup_logging()
    season = "2024"
    logger.info(f"{season}시즌 데이터 로딩 시작")
    
    stat_array, dimension_info = load_data(season)
    logger.info("데이터 로딩 완료")
    
    # 1. 핵심 지표 분석
    core_stats_df = analyze_core_stats(stat_array, dimension_info)
    
    # 2. 월별 성적 추이 분석
    monthly_df = analyze_monthly_trend(stat_array, dimension_info)
    
    # 3. 상황별 성적 분석
    situation_df = analyze_situational_stats(stat_array, dimension_info)
    
    # 4. 시각화
    # 4.1 상관관계 히트맵
    plt.figure(figsize=(10, 8))
    sns.heatmap(core_stats_df[CORE_STATS].corr(), annot=True, cmap='coolwarm', center=0)
    plt.title('핵심 지표 간 상관관계')
    plt.tight_layout()
    plt.savefig('core_stats_correlation.png')
    
    # 4.2 레이더 차트
    fig = go.Figure()
    for player_id in core_stats_df.index:
        values = core_stats_df.loc[player_id][CORE_STATS].values.tolist()
        values.append(values[0])
        
        player_name = PLAYER_NAMES.get(player_id, f"선수 {player_id}")
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=CORE_STATS + [CORE_STATS[0]],
            name=player_name
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, core_stats_df[CORE_STATS].max().max()]
            )),
        showlegend=True,
        title='선수별 핵심 지표 비교'
    )
    fig.write_html('player_comparison.html')
    
    # 4.3 월별 성적 추이 그래프
    plt.figure(figsize=(15, 8))
    for stat in CORE_STATS:
        plt.plot(monthly_df.index.get_level_values(1), monthly_df[stat], label=CORE_STAT_DESCRIPTIONS[stat])
    plt.title('월별 성적 추이')
    plt.xlabel('월')
    plt.ylabel('지표 값')
    plt.legend()
    plt.tight_layout()
    plt.savefig('monthly_trend.png')
    
    # 5. 개별 선수 보고서 생성
    top_players = core_stats_df.nlargest(5, 'war').index
    for player_id in top_players:
        report = create_player_report(player_id, stat_array, dimension_info)
        with open(f'player_report_{player_id}.md', 'w', encoding='utf-8') as f:
            f.write(report)
    
    # 6. 종합 보고서 생성
    with open('analysis_report.md', 'w', encoding='utf-8') as f:
        f.write("""# KBO 선수 성적 분석 종합 보고서

## 1. 개요
이 보고서는 2024 시즌 KBO 선수들의 성적을 다각도로 분석한 결과를 담고 있습니다.

## 2. 분석 방법
- 핵심 성과 지표(KPI) 분석
- 월별 성적 추이 분석
- 상황별 성적 분석
- 선수 간 비교 분석

## 3. 주요 발견점
### 3.1 최고 성과 선수
""")
        
        # WAR 기준 상위 5명 정보 추가
        f.write("\n#### WAR 기준 상위 5명\n")
        top_war = core_stats_df.nlargest(5, 'war')
        f.write(top_war[['player_name'] + CORE_STATS].to_string())
        
        f.write("\n\n### 3.2 지표간 상관관계\n")
        f.write("- core_stats_correlation.png 파일 참조\n")
        
        f.write("\n### 3.3 월별 성적 추이\n")
        f.write("- monthly_trend.png 파일 참조\n")
        
        f.write("\n### 3.4 상황별 성적\n")
        f.write(situation_df.describe().to_string())
        
        f.write("\n\n## 4. 결론 및 제언\n")
        f.write("- 개별 선수 상세 분석은 player_report_*.md 파일들을 참조\n")
        f.write("- 선수간 비교 시각화는 player_comparison.html 파일 참조\n")
    
    logger.info("분석 완료")

if __name__ == "__main__":
    main() 