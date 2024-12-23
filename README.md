# 삼성 라이온즈 선수 분석 프로젝트

## 프로젝트 개요

삼성 라이온즈 선수들의 경기 데이터를 분석하여 성과 지표를 도출하고, 머신러닝 모델을 통해 선수들의 성과를 예측하고 분석 및 유의미한 결과를 도출하는 것을 목표로 했습니다.


### 프로젝트 목표
1. 주요 성과 지표 도출: 선수별 타율, OPS, WAR 등 주요 성과 지표를 분석하여 경기 기여도를 평가.

2. 상황별 성과 분석: 주자 상태, 타석 상황 등 다양한 경기 상황에서 선수들의 성과를 비교.

3. 클러스터링을 통한 그룹화: 유사한 성과 패턴을 가진 선수들을 그룹화하여 팀 전략을 최적화.

4. 머신러닝 모델을 통한 선수 성과 예측: 선수들의 성과를 예측하고 분석 및 유의미한 결과를 도출.

## 프로젝트 구조

```
baseball_analysis/
├── data/
│   ├── raw/                # 원본 데이터
│   ├── processed/          # 전처리된 데이터
│   └── cleaned/           # 정제된 데이터
├── src/
│   ├── collect/           # 데이터 수집 스크립트
│   ├── preprocessing/     # 데이터 전처리 스크립트
│   └── analysis/         # 데이터 분석 스크립트
├── output/
│   ├── player_stats/      # 선수별 통계 분석
│   ├── player_reports_2025/ # 2025 시즌 예측
│   ├── model_analysis/    # 모델 분석 결과
│   ├── models/           # 학습된 모델 저장
│   ├── time_series/      # 시계열 데이터 및 상황별 분석 결과
│   └── clusters/         # 클러스터링 결과
```

## 분석 방법 및 전처리 과정

### 1. 데이터 수집
#### 수집 대상
- **기본 정보**: 선수 ID, 이름, 소속팀
- **경기 기록**: 타율, 출루율, 장타율, WAR 등
- **상황별 기록**: 이닝, 주자, 아웃카운트 등

#### 수집 방법
1. **기본 데이터**
   ```python
   def get_player_data(player_id, split01, split02_1, split02_2=""):
       url = f"http://www.kbreport.com/player/detail/{player_id}"
       # 선수별 기본 기록 수집
   ```

2. **상황별 데이터**
   - **월별 기록**: 2월-11월
   - **요일별 기록**: 월요일-일요일
   - **주자 상황별**:
     - 주자 있음 (onBase)
     - 주자 없음 (emptyBase)
     - 득점권 (onScoring)
     - 만루 (fullBase)
     - 각 루별 상황 (base1, base2, base3 등)
   - **기타 상황**:
     - 홈/원정
     - 투수 유형별
     - 승패별

3. **데이터 저장**
   ```python
   def save_split_tables(player_id, tables):
       # 선수별 디렉토리 생성
       base_directory = f"player_{player_id}_{player_name}"
       # CSV 파일로 저장
   ```

### 2. 데이터 전처리
#### 결측치 처리
- **기본 통계 필드**
  - 경기수, 타석, 타수, 안타, 홈런 등 기본 지표는 0으로 대체
  - 카테고리별 결측치 개별 처리
- **고급 통계 필드**
  - BABIP, OPS, WAR 등은 카테고리별 평균으로 대체
  - 잔여 결측치는 전체 평균 사용
  - 최종 남은 결측치는 0으로 대체

#### 데이터 정제
- **필드 그룹화**
  ```python
  STAT_GROUPS = {
      'basic': ['games', 'plate_appearances', 'at_bats', 'hits', 'home_runs',
                'runs', 'rbis', 'walks', 'strikeouts'],
      'advanced': ['babip', 'avg', 'ops', 'war', 'woba', 'rc', 'wrc']
  }
  ```
- **유효성 검증**
  - 필수 컬럼 존재 확인: player_id, team, category
  - 수치형 데이터 타입 검증
  - 핵심 필드 결측치 확인

### 3. 분석 프로세스


#### 예측 모델링
1. **데이터 준비**
   - 시계열 데이터 구성
   - 특성 스케일링
   - 훈련/검증 세트 분리

2. **모델 학습**
   - Gradient Boosting
     - 하이퍼파라미터 최적화
     - 교차 검증
   - Random Forest
     - 특성 중요도 분석
     - 앙상블 크기 조정
   - XGBoost
     - 조기 종료 설정
     - 특성 선택

3. **모델 평가**
   - RMSE, MAE 계산
   - 예측 신뢰구간 산출
   - 모델 성능 비교

### 4. 시각화 및 보고서 생성
- **성과 지표 시각화**
  - 시계열 트렌드 차트
  - 상관관계 히트맵
  - 분포도 분석

- **예측 결과 시각화**
  - 실제 vs 예측 비교
  - 특성 중요도 차트
  - 신뢰구간 플롯

## 데이터 분석 결과

### 1. 모델 분석 (`/output/model_analysis/`)

#### Gradient Boosting
- **평균 중요도**: 안타 (0.0125), 타수 (0.0023), BABIP (0.0083)
- **OPS 중요도**: 홈런 (0.0378), 볼넷 (0.0272), 안타 (0.0254)
- **WAR 중요도**: 타석 (0.315), 볼넷 (0.165), 안타 (0.163)
- **시각화**: `/output/model_analysis/gb_avg_importance.html`

#### Random Forest
- **평균 중요도**: 안타 (0.1487), 타수 (0.0613), 단타 (0.1133)
- **OPS 중요도**: 홈런 (0.1425), 안타 (0.0997), 볼넷 (0.0695)
- **WAR 중요도**: 타석 (0.214), 안타 (0.1867), 볼넷 (0.1188)
- **시각화**: `/output/model_analysis/rf_avg_importance.html`

#### XGBoost
- **평균 중요도**: 안타 (0.1967), 타석 (0.112), 단타 (0.1484)
- **OPS 중요도**: 홈런 (0.358), 안타 (0.0206), 2루타 (0.0442)
- **WAR 중요도**: 타석 (0.167), 볼넷 (0.181), 안타 (0.269)
- **시각화**: `/output/model_analysis/xgb_avg_importance.html`

### 2. 상황별 분석 (`/output/time_series/`)

#### 주자 상황별 분석
  - **wRC (가중 득점 생성)**
    - 주자 없는 상황과 주자 있는 상황에서 높은 wRC 분포
    - 만루 상황에서는 상대적으로 낮은 wRC 분포로 득점 기회 활용 제한적
  - **RBIs (타점)**
    - 주자 있는 상황과 득점권에서 타점 분포 크게 증가
    - 주자 없는 상황에서는 타점 생성 미미
  - **OPS (출루율 + 장타율)**
    - 주자 있는 상황과 득점권에서 평균적으로 높은 OPS
    - 1루, 2루 주자 상황에서도 준수한 OPS 분포
  - **홈런**
    - 주자 없는 상황에서 가장 높은 발생률
    - 주자 있는 상황에서는 상대적으로 낮은 발생률
  - **2루타**
    - 주자 없는 상황과 주자 있는 상황에서 높은 발생 빈도
    - 1,3루와 2,3루 상황에서는 매우 낮은 발생 빈도
  - **타율**
    - 3루 주자와 2,3루 상황에서 상대적으로 높은 타율 분포
    - 주자 없는 상황에서 가장 넓은 타율 분포


### 3. 클러스터링 분석 (`/output/clusters/`)

#### K-means 클러스터링
- **분석 방법**
  ```python
  from sklearn.cluster import KMeans
  kmeans = KMeans(n_clusters=4, random_state=42)
  clusters = kmeans.fit_predict(scaled_features)
  ```
- **클러스터 특성**
  - **클러스터 1 (파워 히터)**
    - 높은 장타율과 홈런 생산력
    - OPS 0.800 이상
    - 대표 선수: 구자욱
  - **클러스터 2 (컨택 히터)**
    - 높은 타율과 출루율
    - 낮은 삼진율
    - 대표 선수: 김지찬
  - **클러스터 3 (균형 타자)**
    - 안정적인 타격 지표
    - 중상위권 WAR
    - 대표 선수: 강민호

#### 시각화
- **2D 클러스터링**: `output/clusters/cluster_overall_2d.png`
  - X축: OPS, Y축: WAR
  - 클러스터별 색상 구분
- **3D 클러스터링**: `output/clusters/cluster_overall_3d.png`
  - X축: 타율, Y축: 장타율, Z축: WAR
  - 클러스터별 산점도


### 4. 주요 선수 상세 분석 (`/output/player_stats/`)

#### 주요 선수 기록
| 선수명(ID) | 경기 | 출장률(%) | 타율 | OPS | WAR | 안타 | 2루타 | 도루 | 볼넷 | 출루 |
|------------|-------|-----------|------|------|-----|------|-------|------|------|------|
| 강민호(738) | 129 | 89.6 | 0.269 | 0.788 | 1.7 | 71 | 22 | 0 | 29 | 96 |
| 김지찬(1754) | 135 | 93.8 | 0.232 | 0.573 | -0.7 | 13 | 1 | 21 | 24 | 44 |
| 구자욱(517) | 116 | 80.6 | 0.349 | 0.951 | 4.0 | 57 | 11 | 17 | 45 | 79 |

### 5. 2025 시즌 선수 성적 예측 (`/output/player_reports_2025/`)

#### 주요 선수별 예측 리포트
- **예측 결과**: 
  - 강민호: `/output/player_reports_2025/prediction_summary/player_738_2025_prediction.html`
  - 김지찬: `/output/player_reports_2025/prediction_summary/player_1754_2025_prediction.html`
  - 구자욱: `/output/player_reports_2025/prediction_summary/player_517_2025_prediction.html`
   


## 설치 및 실행

1. 필요 패키지 설치
```bash
pip install -r requirements.txt
```

2. 데이터 수집
```bash
python src/collect/main.py
```

3. 데이터 전처리
```bash
python src/preprocessing/clean_player_data.py
```

4. 분석 실행
```bash
python src/analysis/time_series.py
```

## 주요 발견사항

1. **성과 지표 영향 요인**
   - 안타와 타석이 성과 지표에 가장 큰 영향
   - WAR은 볼넷과 홈런에 높은 의존도

2. **상황별 특성**
   - 주자 상황에 따라 RC/27 변동 큼

3. **선수별 특성**
   - 구자욱: 높은 타율과 OPS
   - 강민호: 안정적인 중장거리 타격
   - 김지찬: 높은 출장률과 도루

## 데이터 출처

KBO 리포트: https://www.kbreport.com/
