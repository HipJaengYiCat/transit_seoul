"""
코드 2: DuckDB + Arrow 최적화 방식
- 병렬 처리 제한을 통한 메모리 경합 방지
- Arrow 형식을 활용한 메모리 효율성 향상
- 성능 모니터링 및 단계별 최적화 적용
"""

import os
import time
import psutil
import duckdb
import pandas as pd
import numpy as np
import folium
import geopandas as gpd
import branca.colormap as cm
from datetime import datetime
from pyproj import Transformer
import warnings
warnings.filterwarnings('ignore')

class PerformanceMonitor:
    """성능 모니터링 클래스"""
    
    def __init__(self):
        self.START_TIME = None
        self.START_MEMORY = None
        self.LAST_CHECKPOINT_TIME = None
        self.PROCESS = psutil.Process(os.getpid())
        self.CHECKPOINTS = []
        
    def start(self):
        """모니터링 시작"""
        self.START_TIME = time.time()
        self.LAST_CHECKPOINT_TIME = self.START_TIME
        self.START_MEMORY = self.PROCESS.memory_info().rss / 1024 / 1024  # MB
        print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"시작 메모리: {self.START_MEMORY:.2f} MB")
        print("="*60)
        
    def checkpoint(self, STEP_NAME):
        """중간 체크포인트"""
        CURRENT_TIME = time.time()
        CURRENT_MEMORY = self.PROCESS.memory_info().rss / 1024 / 1024  # MB
        
        TOTAL_ELAPSED = CURRENT_TIME - self.START_TIME
        STEP_TIME = CURRENT_TIME - self.LAST_CHECKPOINT_TIME
        
        CHECKPOINT_DATA = {
            'step': STEP_NAME,
            'step_time': STEP_TIME,
            'total_elapsed': TOTAL_ELAPSED,
            'memory_usage': CURRENT_MEMORY,
            'memory_increase': CURRENT_MEMORY - self.START_MEMORY
        }
        self.CHECKPOINTS.append(CHECKPOINT_DATA)
        
        print(f"[{STEP_NAME}]")
        print(f"단계 실행시간: {STEP_TIME:.2f}초")
        print(f"총 경과시간: {TOTAL_ELAPSED:.2f}초")
        print(f"현재 메모리: {CURRENT_MEMORY:.2f} MB ({CURRENT_MEMORY/1024:.2f} GB)")
        print(f"메모리 증가: {CURRENT_MEMORY - self.START_MEMORY:.2f} MB")
        print("-" * 40)
        
        self.LAST_CHECKPOINT_TIME = CURRENT_TIME
        
    def end(self):
        """모니터링 종료 및 리포트"""
        END_TIME = time.time()
        END_MEMORY = self.PROCESS.memory_info().rss / 1024 / 1024  # MB
        TOTAL_TIME = END_TIME - self.START_TIME
        MEMORY_USAGE = END_MEMORY - self.START_MEMORY
        
        print("="*60)
        print("최종 성능 리포트")
        print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"총 실행 시간: {TOTAL_TIME:.2f}초 ({TOTAL_TIME/60:.2f}분)")
        print(f"메모리 사용량: {MEMORY_USAGE:.2f} MB ({MEMORY_USAGE/1024:.2f} GB)")
        print("="*60)
        
        return {
            'total_time': TOTAL_TIME,
            'memory_usage_mb': MEMORY_USAGE,
            'memory_usage_gb': MEMORY_USAGE/1024
        }

def setup_duckdb_optimization():
    """DuckDB 최적화 설정"""
    duckdb.sql("PRAGMA threads=1")  # 병렬 처리 제한으로 메모리 경합 방지
    duckdb.sql("PRAGMA memory_limit='512MB'")  # 메모리 제한 설정
    print("DuckDB 최적화 설정 완료")

def load_month_data_optimized(YEAR, MONTH, MONITOR=None):
    """Arrow 방식을 적용한 최적화된 데이터 로딩"""
    QUERY = f"""
        SELECT 
            DATE, O_ADMI_CD, O_CELL_ID, O_CELL_X, O_CELL_Y, O_CELL_TP,
            D_ADMI_CD, D_CELL_ID, D_CELL_X, D_CELL_Y, D_CELL_TP,
            MOVE_PURPOSE, MOVE_DIST, MOVE_TIME, TOTAL_CNT
        FROM '/home1/rldnjs16/transit/dataset/data_month/year={YEAR:04}/month={MONTH:02}/data.parquet'
    """
    
    print(f"{MONTH}월 데이터 로딩 중...")
    
    try:
        # Arrow 테이블로 먼저 로드 (메모리 효율적)
        TABLE = duckdb.query(QUERY).arrow()
        
        # pandas 변환 시 최적화 옵션 적용
        DF = TABLE.to_pandas(
            split_blocks=True,    # 메모리 블록 분할로 효율성 향상
            self_destruct=True    # Arrow 테이블 즉시 해제
        )
        
        print(f"데이터 로드 완료: {len(DF):,}행 x {len(DF.columns)}열")
        
        if MONITOR:
            MONITOR.checkpoint(f"DuckDB {MONTH}월 데이터 로딩")
        
        return DF
        
    except Exception as e:
        print(f"{YEAR}-{MONTH} 데이터 로드 실패: {e}")
        return None

def load_multiple_months(YEAR_MONTH_LIST, MONITOR=None):
    """여러 달 데이터를 순차적으로 로드"""
    DATAFRAMES = []
    
    for YEAR, MONTH in YEAR_MONTH_LIST:
        DF_MONTH = load_month_data_optimized(YEAR, MONTH, MONITOR)
        if DF_MONTH is not None:
            DATAFRAMES.append(DF_MONTH)
        else:
            print(f"{YEAR}-{MONTH} 데이터 스킵")
    
    if not DATAFRAMES:
        raise Exception("로드된 데이터가 없습니다.")
    
    # 전체 데이터 결합
    DF_TOTAL = pd.concat(DATAFRAMES, ignore_index=True)
    
    if MONITOR:
        MONITOR.checkpoint("전체 데이터 결합 완료")
    
    print(f"전체 데이터 결합 완료: {len(DF_TOTAL):,}행")
    return DF_TOTAL

def filter_and_aggregate_data(DF, MONITOR=None):
    """데이터 필터링 및 그룹핑"""
    # 목적별 필터링 (MOVE_PURPOSE = 1)
    DF_FILTERED = DF[DF['MOVE_PURPOSE'] == 1]
    print(f"필터링 후 데이터: {len(DF_FILTERED):,}행")
    
    # 격자별 집계
    DF_GROUPED = DF_FILTERED.groupby('O_CELL_ID').agg({
        'MOVE_DIST': 'sum',
        'MOVE_TIME': 'sum',
        'TOTAL_CNT': 'sum',
        'O_CELL_X': 'first',
        'O_CELL_Y': 'first'
    }).reset_index()
    
    if MONITOR:
        MONITOR.checkpoint("데이터 필터링 및 그룹핑")
    
    print(f"집계 후 격자 수: {len(DF_GROUPED):,}개")
    return DF_GROUPED

def convert_coordinates(DF, MONITOR=None):
    """좌표 변환 (EPSG:5179 -> WGS84)"""
    TRANSFORMER = Transformer.from_crs("epsg:5179", "epsg:4326", always_xy=True)
    
    DF['LON'], DF['LAT'] = TRANSFORMER.transform(
        DF['O_CELL_X'].values,
        DF['O_CELL_Y'].values
    )
    
    if MONITOR:
        MONITOR.checkpoint("좌표 변환 (EPSG:5179 -> WGS84)")
    
    print("좌표 변환 완료")
    return DF

def create_visualization_map(DF, OUTPUT_PATH, MONITOR=None):
    """지도 시각화 생성"""
    # 로그 스케일 적용
    DF['MOVE_TIME_LOG'] = np.log1p(DF['MOVE_TIME'])
    MIN_T = DF['MOVE_TIME_LOG'].min()
    MAX_T = DF['MOVE_TIME_LOG'].max()
    
    # 색상 맵 생성
    COLORMAP = cm.linear.YlOrRd_09.scale(MIN_T, MAX_T)
    COLORMAP.caption = "이동시간 (분, 로그스케일)"
    
    # 지도 객체 생성
    M = folium.Map(location=[37.5665, 126.9780], zoom_start=10)
    COLORMAP.add_to(M)
    
    if MONITOR:
        MONITOR.checkpoint("지도 객체 생성 및 색상 설정")
    
    # 원 추가
    for _, ROW in DF.iterrows():
        folium.Circle(
            location=[ROW['LAT'], ROW['LON']],
            radius=max(ROW['TOTAL_CNT'] / 25, 5),
            color=COLORMAP(ROW['MOVE_TIME_LOG']),
            fill=True,
            fill_opacity=0.7,
            popup=(
                f"격자ID: {ROW['O_CELL_ID']}<br>"
                f"좌표: ({ROW['O_CELL_X']}, {ROW['O_CELL_Y']})<br>"
                f"이동인구수: {ROW['TOTAL_CNT']:,}명<br>"
                f"이동시간: {ROW['MOVE_TIME']:,.0f}분"
            )
        ).add_to(M)
    
    # 지도 저장
    M.save(OUTPUT_PATH)
    
    if MONITOR:
        MONITOR.checkpoint("지도 시각화 생성 및 저장")
    
    print(f"지도 저장 완료: {OUTPUT_PATH}")
    return M

def create_admin_level_analysis(DF, MONITOR=None):
    """행정동 단위 분석"""
    # 행정동별 집계
    DF_ADMIN = DF.groupby('O_ADMI_CD').agg({
        'MOVE_DIST': 'sum',
        'MOVE_TIME': 'sum',
        'TOTAL_CNT': 'sum',
        'O_CELL_X': 'mean',
        'O_CELL_Y': 'mean'
    }).reset_index()
    
    # 좌표 변환
    DF_ADMIN = convert_coordinates(DF_ADMIN)
    
    if MONITOR:
        MONITOR.checkpoint("행정동 단위 집계 및 좌표 변환")
    
    print(f"행정동 수: {len(DF_ADMIN):,}개")
    return DF_ADMIN

def main():
    """메인 실행 함수"""
    # 성능 모니터링 시작
    MONITOR = PerformanceMonitor()
    MONITOR.start()
    
    # DuckDB 최적화 설정
    setup_duckdb_optimization()
    
    # 분석 대상 기간 설정 (예: 2024년 6월~12월)
    YEAR_MONTH_LIST = [
        (2024, 6), (2024, 7), (2024, 8), 
        (2024, 9), (2024, 10), (2024, 11), (2024, 12)
    ]
    
    try:
        # 1. 데이터 로딩
        DF_RAW = load_multiple_months(YEAR_MONTH_LIST, MONITOR)
        
        # 2. 데이터 필터링 및 집계
        DF_AGGREGATED = filter_and_aggregate_data(DF_RAW, MONITOR)
        
        # 원본 데이터 메모리 해제
        del DF_RAW
        
        # 3. 좌표 변환
        DF_FINAL = convert_coordinates(DF_AGGREGATED, MONITOR)
        
        # 4. 격자 단위 지도 생성
        GRID_MAP_PATH = "/home1/rldnjs16/transit/map_visualization/seoul_transit_grid_optimized.html"
        create_visualization_map(DF_FINAL, GRID_MAP_PATH, MONITOR)
        
        # 5. 행정동 단위 분석
        DF_ADMIN = create_admin_level_analysis(DF_FINAL, MONITOR)
        
        # 6. 행정동 단위 지도 생성
        ADMIN_MAP_PATH = "/home1/rldnjs16/transit/map_visualization/seoul_transit_admin_optimized.html"
        create_visualization_map(DF_ADMIN, ADMIN_MAP_PATH, MONITOR)
        
        # 7. 최종 성능 리포트
        PERFORMANCE_RESULTS = MONITOR.end()
        
        # 결과 요약
        print("\n" + "="*60)
        print("분석 완료 요약")
        print("="*60)
        print(f"처리된 기간: {len(YEAR_MONTH_LIST)}개월")
        print(f"총 격자 수: {len(DF_FINAL):,}개")
        print(f"총 행정동 수: {len(DF_ADMIN):,}개")
        print(f"총 이동인구: {DF_FINAL['TOTAL_CNT'].sum():,}명")
        print(f"평균 이동시간: {DF_FINAL['MOVE_TIME'].mean():.1f}분")
        print(f"처리 시간: {PERFORMANCE_RESULTS['total_time']:.1f}초")
        print(f"메모리 사용량: {PERFORMANCE_RESULTS['memory_usage_gb']:.2f} GB")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("코드 2: DuckDB + Arrow 최적화 방식 시작")
    SUCCESS = main()
    
    if SUCCESS:
        print("\n성공")
    else:
        print("\n오류")