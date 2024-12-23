import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 상수 정의
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
CLEANED_DIR = DATA_DIR / "cleaned/player_stats"
OUTPUT_DIR = BASE_DIR / "output/clusters"

# 분석할 특성 그룹 정의
FEATURE_GROUPS = {
    'overall': {
        'features': ['war', 'wrc', 'rc27'],
        'n_clusters': 4,
        'description': '종합 능력'
    }
}

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def load_player_data() -> pd.DataFrame:
    """정제된 선수 데이터 로드"""
    all_data = []
    for file_path in CLEANED_DIR.glob("player_*.csv"):
        df = pd.read_csv(file_path)
        # total 카테고리 데이터만 선택
        total_data = df[df['category'] == 'total'].copy()
        if not total_data.empty:
            all_data.append(total_data)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # 데이터 전처리
    logger = logging.getLogger(__name__)
    logger.info(f"원본 데이터 형태: {combined_data.shape}")
    
    # NaN 값이 있는 행 제거
    combined_data = combined_data.dropna(subset=['player_id', 'team'])
    
    # 수치형 컬럼의 NaN을 0으로 대체
    numeric_columns = combined_data.select_dtypes(include=[np.number]).columns
    combined_data[numeric_columns] = combined_data[numeric_columns].fillna(0)
    
    logger.info(f"전처리 후 데이터 형태: {combined_data.shape}")
    logger.info("결측값 현황:")
    logger.info(combined_data.isna().sum().to_string())
    
    return combined_data

def perform_clustering(data: pd.DataFrame, features: list, n_clusters: int) -> tuple:
    """클러스터링 수행"""
    logger = logging.getLogger(__name__)
    
    # 특성 데이터 추출
    X = data[features].copy()
    
    # 결측값 확인 및 처리
    if X.isna().any().any():
        logger.warning("클러스터링 특성에 결측값이 있어 0으로 대체합니다.")
        logger.warning(f"결측값 현황:\n{X.isna().sum()}")
        X = X.fillna(0)
    
    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 클러스터링 수행
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # 클러스터 중심점을 원래 스케일로 변환
    centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # 클러스터링 결과 요약
    logger.info(f"클러스터 크기: {np.bincount(clusters)}")
    
    return clusters, X_scaled, centers_original

def create_cluster_visualization(data: pd.DataFrame, group_name: str, features: list, 
                               clusters: np.ndarray, centers: np.ndarray):
    """클러스터 시각화"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 데이터 준비
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = clusters
    
    # 2D 산점도 (주요 2개 특성)
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(data[features[0]], data[features[1]], 
                         c=clusters, cmap='viridis', alpha=0.6)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, linewidths=3)
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title(f'{FEATURE_GROUPS[group_name]["description"]} 클러스터링')
    plt.colorbar(scatter)
    
    # 선수 이름 표시
    for i, row in data_with_clusters.iterrows():
        plt.annotate(row['player_id'], 
                    (row[features[0]], row[features[1]]),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'cluster_{group_name}_2d.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 특성별 박스플롯
    plt.figure(figsize=(15, 6))
    for i, feature in enumerate(features, 1):
        plt.subplot(1, len(features), i)
        sns.boxplot(x='Cluster', y=feature, data=data_with_clusters)
        plt.title(feature)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'cluster_{group_name}_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plotly를 사용한 3D 산점도 (3개 이상의 특성이 있는 경우)
    if len(features) >= 3:
        fig = px.scatter_3d(
            data_with_clusters,
            x=features[0],
            y=features[1],
            z=features[2],
            color='Cluster',
            hover_data=['player_id', 'team'] + features,
            title=f'{FEATURE_GROUPS[group_name]["description"]} 3D 클러스터링',
            labels={col: col for col in features}
        )
        
        # 중심점 추가
        fig.add_trace(go.Scatter3d(
            x=centers[:, 0],
            y=centers[:, 1],
            z=centers[:, 2],
            mode='markers',
            marker=dict(color='red', size=10, symbol='x'),
            name='Cluster Centers'
        ))
        
        fig.write_html(OUTPUT_DIR / f'cluster_{group_name}_3d.html')

def analyze_clusters(data: pd.DataFrame, clusters: np.ndarray, group_name: str, features: list) -> pd.DataFrame:
    """클러스터 분석"""
    cluster_stats = []
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = clusters
    
    for cluster in range(clusters.max() + 1):
        cluster_data = data_with_clusters[data_with_clusters['Cluster'] == cluster]
        stats = {
            'Cluster': cluster,
            'Size': len(cluster_data),
            'Players': ', '.join(f"{row['player_id']}({row['team']})" 
                               for _, row in cluster_data.iterrows())
        }
        
        # 각 특성의 평균값과 표준편차 계산
        for feature in features:
            stats[f'{feature}_mean'] = cluster_data[feature].mean()
            stats[f'{feature}_std'] = cluster_data[feature].std()
        
        cluster_stats.append(stats)
    
    cluster_summary = pd.DataFrame(cluster_stats)
    
    # 결과 저장
    with pd.ExcelWriter(OUTPUT_DIR / f'cluster_{group_name}_summary.xlsx') as writer:
        # 요약 정보
        cluster_summary.to_excel(writer, sheet_name='Summary', index=False)
        
        # 전체 데이터
        data_with_clusters.to_excel(writer, sheet_name='Full_Data', index=False)
    
    return cluster_summary

def main():
    logger = setup_logging()
    
    try:
        # 데이터 로드
        logger.info("선수 데이터 로드 중...")
        data = load_player_data()
        
        # 각 특성 그룹별 클러스터링
        for group_name, group_info in FEATURE_GROUPS.items():
            logger.info(f"{group_info['description']} 클러스터링 수행 중...")
            
            features = group_info['features']
            n_clusters = group_info['n_clusters']
            
            # 클러스터링 수행
            clusters, scaled_data, centers = perform_clustering(
                data, features, n_clusters
            )
            
            # 시각화
            logger.info(f"{group_info['description']} 시각화 생성 중...")
            create_cluster_visualization(
                data, group_name, features, clusters, centers
            )
            
            # 클러스터 분석
            logger.info(f"{group_info['description']} 클러스터 분석 중...")
            cluster_summary = analyze_clusters(
                data, clusters, group_name, features
            )
            
            logger.info(f"{group_info['description']} 클러스터 요약:")
            for _, row in cluster_summary.iterrows():
                logger.info(f"클러스터 {row['Cluster']}: {row['Size']}명의 선수")
        
        logger.info("모든 분석 완료")
        
    except Exception as e:
        logger.error(f"분석 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 