import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List

# 상수 정의
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed/player_stats"
CLEANED_DIR = DATA_DIR / "cleaned/player_stats"

# 통계 필드 그룹 정의 (최소화)
STAT_GROUPS = {
    'basic': [
        'games', 'plate_appearances', 'at_bats', 'hits', 'home_runs',
        'runs', 'rbis', 'walks', 'strikeouts'
    ],
    'advanced': [
        'babip', 'avg', 'ops', 'war', 'woba', 'rc', 'wrc'
    ]
}

# 제거할 컬럼 (100% 결측치)
COLUMNS_TO_DROP = ['obp', 'slg']

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def analyze_missing_values(df: pd.DataFrame) -> Dict:
    """결측치 분석"""
    missing_info = {
        'missing_counts': df.isnull().sum().to_dict(),
        'missing_percentages': (df.isnull().sum() / len(df) * 100).to_dict()
    }
    return missing_info

def handle_missing_values(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """결측치 처리"""
    df = df.copy()
    
    # 카테고리 관련 결측치 처리
    df['subcategory'] = df.apply(
        lambda row: 'total' if row['category'] == 'total' and pd.isna(row['subcategory']) 
        else row['subcategory'],
        axis=1
    )
    
    # 데이터 타입 변환
    for group, fields in STAT_GROUPS.items():
        for field in fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')
    
    # 통계 그룹별 결측치 처리
    for group, fields in STAT_GROUPS.items():
        logger.info(f"{group} 그룹의 결측치 처리 중...")
        available_fields = [f for f in fields if f in df.columns]
        
        if group == 'basic':
            # 기본 통계는 0으로 대체
            df[available_fields] = df[available_fields].fillna(0)
        
        elif group == 'advanced':
            # 고급 지표는 카테고리별 평균으로 대체
            for field in available_fields:
                # 카테고리별 평균 계산
                category_means = df.groupby('category')[field].transform('mean')
                df[field] = df[field].fillna(category_means)
                
                # 여전히 남아있는 NaN은 전체 평균으로 대체
                total_mean = df[field].mean()
                df[field] = df[field].fillna(total_mean)
                
                # 마지막으로 남은 NaN은 0으로 대체
                df[field] = df[field].fillna(0)
    
    return df

def validate_data(df: pd.DataFrame) -> bool:
    """데이터 유효성 검사"""
    # 필수 컬럼 확인
    required_columns = ['player_id', 'team', 'category']
    if not all(col in df.columns for col in required_columns):
        return False
    
    # 데이터 타입 확인
    numeric_columns = sum(STAT_GROUPS.values(), [])
    for col in numeric_columns:
        if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
            return False
    
    # NaN 값 확인
    if df[required_columns].isnull().any().any():
        return False
    
    return True

def clean_player_data(input_file: Path, output_file: Path, logger: logging.Logger) -> bool:
    """선수 데이터 정제"""
    try:
        # 데이터 로드
        df = pd.read_csv(input_file)
        logger.info(f"파일 로드 완료: {input_file.name}")
        
        # 결측치 분석
        missing_info = analyze_missing_values(df)
        logger.info("결측치 분석 결과:")
        for col, pct in missing_info['missing_percentages'].items():
            if pct > 0:
                logger.info(f"- {col}: {pct:.2f}% 결측")
        
        # 제거할 컬럼 처리
        columns_to_drop = [col for col in COLUMNS_TO_DROP if col in df.columns]
        if columns_to_drop:
            logger.info(f"제거할 컬럼: {columns_to_drop}")
            df = df.drop(columns=columns_to_drop)
        
        # 결측치 처리
        df = handle_missing_values(df, logger)
        
        # 데이터 유효성 검사
        if not validate_data(df):
            logger.error("데이터 유효성 검사 실패")
            return False
        
        # 최종 컬럼 확인
        logger.info(f"최종 컬럼 목록: {df.columns.tolist()}")
        
        # 정제된 데이터 저장
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.info(f"정제된 데이터 저장 완료: {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"데이터 정제 중 오류 발생: {str(e)}")
        return False

def main():
    logger = setup_logging()
    
    try:
        # 처리할 파일 목록 가져오기
        input_files = list(PROCESSED_DIR.glob("player_*.csv"))
        logger.info(f"처리할 파일 수: {len(input_files)}")
        
        success_count = 0
        for input_file in input_files:
            output_file = CLEANED_DIR / input_file.name
            if clean_player_data(input_file, output_file, logger):
                success_count += 1
        
        logger.info(f"처리 완료: {success_count}/{len(input_files)} 파일 성공")
        
    except Exception as e:
        logger.error(f"처리 중 오류 발생: {str(e)}")
        raise

if __name__ == "__main__":
    main() 