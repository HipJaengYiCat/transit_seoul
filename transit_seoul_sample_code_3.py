#%%
"""
메모리 최적화된 대용량 이동 데이터 분석 (수정버전)
- 배치 처리를 통한 메모리 사용량 제어
- 월별 ADMI 매핑 적용
- 통합 HTML 대시보드 생성
- 행정동별 choropleth 시각화
- 다양한 기간별 분석 지원
"""

import os
import gc
import time
import psutil
import folium 
import duckdb
import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
import branca.colormap as cm
import matplotlib.pyplot as plt

from folium import plugins
from datetime import datetime
from pyproj import Transformer
from branca.colormap import linear
from folium.plugins import HeatMap
from shapely.geometry import Polygon, MultiPolygon

import warnings
warnings.filterwarnings('ignore')

now = datetime.now()
filename_time = now.strftime("%Y-%m-%d_%H-%M-%S")

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.last_checkpoint_time = None
        self.process = psutil.Process(os.getpid())
        self.checkpoints = []
        
    def start(self):
        """모니터링 시작"""
        self.start_time = time.time()
        self.last_checkpoint_time = self.start_time
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"시작 메모리: {self.start_memory:.2f} MB")
        print("="*60)
        
    def checkpoint(self, step_name):
        """메모리 사용량 포함 체크포인트"""
        current_time = time.time()
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        total_elapsed = current_time - self.start_time
        step_time = current_time - self.last_checkpoint_time
        
        checkpoint_data = {
            'step': step_name,
            'step_time': step_time,
            'total_elapsed': total_elapsed,
            'memory_usage': current_memory,
            'memory_increase': current_memory - self.start_memory
        }
        self.checkpoints.append(checkpoint_data)
        
        print(f"[{step_name}]")
        print(f"단계 실행시간: {step_time:.2f}초")
        print(f"총 경과시간: {total_elapsed:.2f}초")
        print(f"현재 메모리: {current_memory:.2f} MB ({current_memory/1024:.2f} GB)")
        print(f"메모리 증가: {current_memory - self.start_memory:.2f} MB")
        
        # 메모리 사용량이 너무 높으면 경고
        if current_memory > 8 * 1024:  # 8GB 이상
            print(f"메모리 사용량 높음: {current_memory/1024:.1f} GB")
        
        print("-" * 40)
        self.last_checkpoint_time = current_time
        
    def end(self):
        """최종 리포트"""
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024
        total_time = end_time - self.start_time
        memory_usage = end_memory - self.start_memory
        
        print("="*60)
        print("최종 성능 리포트")
        print(f"종료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"총 실행 시간: {total_time:.2f}초 ({total_time/60:.2f}분)")
        print(f"메모리 사용량: {memory_usage:.2f} MB ({memory_usage/1024:.2f} GB)")
        print("="*60)
        
        return {
            'total_time': total_time,
            'memory_usage_mb': memory_usage,
            'memory_usage_gb': memory_usage/1024
        }

def monitor_memory_usage():
    """현재 메모리 사용량 체크"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    memory_gb = memory_mb / 1024
    
    print(f"현재 메모리 사용량: {memory_mb:.1f} MB ({memory_gb:.2f} GB)")
    
    # 시스템 전체 메모리 정보
    system_memory = psutil.virtual_memory()
    print(f"시스템 메모리 사용률: {system_memory.percent:.1f}%")
    print(f"사용 가능한 메모리: {system_memory.available / 1024**3:.1f} GB")
    
    return memory_gb

def force_memory_cleanup():
    """강제 메모리 정리"""
    gc.collect()
    time.sleep(0.1)

def load_admi_data(year, month):
    """해당 월의 ADMI 데이터 로드"""
    admi_file = f'/home1/rldnjs16/transit_seoul/dataset/ADMI_RE/ADMI_{year}{month:02d}.csv'
    
    if os.path.exists(admi_file):
        df_admi = pd.read_csv(admi_file)
        df_admi['ADMI_CD'] = df_admi['ADMI_CD'].astype(str)
        print(f"ADMI 데이터 로드 완료: {admi_file}")
        return df_admi
    else:
        print(f"ADMI 파일이 존재하지 않음: {admi_file}")
        return None

def load_and_aggregate_batch(year_months, batch_size=3, save_intermediate=True):
    """
    배치별로 데이터를 로드하고 집계하여 메모리 사용량 제어
    """
    monitor = PerformanceMonitor()
    monitor.start()
    
    # DuckDB 설정 최적화
    duckdb.sql("PRAGMA threads=2")
    duckdb.sql("PRAGMA memory_limit='512MB'")
    
    final_result = None
    
    # 배치별 처리
    for i in range(0, len(year_months), batch_size):
        batch = year_months[i:i+batch_size]
        print(f"\n{'='*50}")
        print(f"배치 {i//batch_size + 1} 처리 중: {batch}")
        print(f"{'='*50}")
        
        batch_data = []
        
        # 배치 내 데이터 로드 및 즉시 필터링
        for year, month in batch:
            print(f"{year}-{month:02d} 데이터 처리 중...")
            
            query = f"""
                SELECT 
                    O_ADMI_CD, O_CELL_ID, O_CELL_X, O_CELL_Y,
                    MOVE_DIST, MOVE_TIME, TOTAL_CNT
                FROM '/home1/rldnjs16/transit_seoul/dataset/data_month/year={year:04}/month={month:02}/SEOUL_PURPOSE_250M_IN_{year:04}{month:02}.parquet'
                WHERE MOVE_PURPOSE = 1
            """
            
            try:
                table = duckdb.query(query).arrow()
                df_month = table.to_pandas(split_blocks=True, self_destruct=True)
                
                # 즉시 집계하여 메모리 사용량 줄이기
                df_month_agg = df_month.groupby(['O_ADMI_CD', 'O_CELL_ID']).agg({
                    'MOVE_DIST': 'sum',
                    'MOVE_TIME': 'sum', 
                    'TOTAL_CNT': 'sum',
                    'O_CELL_X': 'first',
                    'O_CELL_Y': 'first'
                }).reset_index()
                
                batch_data.append(df_month_agg)
                
                # 원본 데이터 즉시 삭제
                del df_month, table
                force_memory_cleanup()
                
                print(f"  → 집계 완료: {len(df_month_agg):,}행")
                
            except Exception as e:
                print(f"{year}-{month:02d} 데이터 로드 실패: {e}")
                continue
        
        if not batch_data:
            print("배치에서 처리된 데이터가 없습니다.")
            continue
            
        # 배치 내 데이터 결합 및 재집계
        batch_combined = pd.concat(batch_data, ignore_index=True)
        del batch_data
        force_memory_cleanup()
        
        monitor.checkpoint(f"배치 {i//batch_size + 1} 데이터 로드 완료")
        
        # 배치별 최종 집계
        batch_result = batch_combined.groupby(['O_ADMI_CD', 'O_CELL_ID']).agg({
            'MOVE_DIST': 'sum',
            'MOVE_TIME': 'sum',
            'TOTAL_CNT': 'sum', 
            'O_CELL_X': 'first',
            'O_CELL_Y': 'first'
        }).reset_index()
        
        del batch_combined
        force_memory_cleanup()
        
        monitor.checkpoint(f"배치 {i//batch_size + 1} 집계 완료")
        
        # 중간 결과 저장
        if save_intermediate:
            batch_file = f"/home1/rldnjs16/transit_seoul/temp/batch_{i//batch_size + 1}.parquet"
            os.makedirs(os.path.dirname(batch_file), exist_ok=True)
            batch_result.to_parquet(batch_file)
            print(f"중간 결과 저장: {batch_file}")
        
        # 전체 결과와 결합
        if final_result is None:
            final_result = batch_result
        else:
            combined = pd.concat([final_result, batch_result], ignore_index=True)
            del final_result, batch_result
            force_memory_cleanup()
            
            final_result = combined.groupby(['O_ADMI_CD', 'O_CELL_ID']).agg({
                'MOVE_DIST': 'sum',
                'MOVE_TIME': 'sum',
                'TOTAL_CNT': 'sum',
                'O_CELL_X': 'first', 
                'O_CELL_Y': 'first'
            }).reset_index()
            
            del combined
            force_memory_cleanup()
        
        monitor.checkpoint(f"배치 {i//batch_size + 1} 전체 결과 통합 완료")
        print(f"현재까지 총 집계 결과: {len(final_result):,}행")
    
    performance_results = monitor.end()
    return final_result, performance_results

def create_geojson_visualization(df_g_h_g, year, month, output_path):
    """월별 행정동 choropleth 지도 생성 (원래 코드 방식 유지)"""
    
    # 입력 데이터 확인
    print(f"입력 데이터 컬럼: {df_g_h_g.columns.tolist()}")
    print(f"입력 데이터 크기: {len(df_g_h_g)}")
    
    try:
        # 행정동 GeoJSON 로딩
        gdf_adm = gpd.read_file("/home1/rldnjs16/transit_seoul/dataset/Administrative_boundaries/BND_ADM_DONG_PG/BND_ADM_DONG_PG.shp")
        
        # 지역명 전체 정보 로딩
        gdf = pd.read_excel(
            "/home1/rldnjs16/transit_seoul/dataset/Administrative_boundaries/센서스 공간정보 지역 코드.xlsx",
            skiprows=1,
            engine='openpyxl',
            dtype={'시도코드': str, '시군구코드': str, '읍면동코드': str}
        )
        
        gdf = gdf[(gdf['시도명칭'] == '서울특별시') | (gdf['시도명칭'] == '경기도') | (gdf['시도명칭'] == '인천광역시')]
        gdf['NAME'] = gdf['시도코드'] + gdf['시군구코드'] + gdf['읍면동코드']
        
        # Merge
        gdf_adm['ADM_CD'] = gdf_adm['ADM_CD'].astype(str)
        gdf_merge = gdf_adm.merge(gdf, left_on='ADM_CD', right_on='NAME', how='left')
        gdf_merge_ = gdf_merge[~gdf_merge['시도코드'].isna()]
        gdf_merge_['FULL_NAME'] = gdf_merge_['시도명칭'] + ' ' + gdf_merge_['시군구명칭'] + ' ' + gdf_merge_['읍면동명칭']
        
        gdf_merge_.drop(columns=['시도명칭', '시군구명칭', '읍면동명칭', 'BASE_DATE', 'NAME', '시도코드', '시군구코드', '읍면동코드'], inplace=True)
        
        col_list = ['ADM_CD', 'ADM_NM', 'FULL_NAME']
        col_r = [col for col in gdf_merge_.columns if col not in col_list]
        gdf_merge_ = gdf_merge_[col_list + col_r]
        
        # 해당 월의 행정동 코드 매칭
        df_admi = load_admi_data(year, month)
        if df_admi is None:
            # 기본으로 6월 데이터 사용
            df_admi = load_admi_data(2024, 6)
        
        if df_admi is not None:
            df_admi['ADMI_CD'] = df_admi['ADMI_CD'].astype(str)
            gdf_merge_2 = gdf_merge_.merge(df_admi, left_on='FULL_NAME', right_on='FULL_NM', how='left')
            
            # 필요한 컬럼만 유지
            gdf_merge_2 = gdf_merge_2[['FULL_NM', 'ADMI_CD', 'geometry']].dropna(subset=['ADMI_CD'])
            
            # 최종 merge
            df_g_h_g_copy = df_g_h_g.copy()
            df_g_h_g_copy['O_ADMI_CD'] = df_g_h_g_copy['O_ADMI_CD'].astype(str)
            gdf_merge_2['ADMI_CD'] = gdf_merge_2['ADMI_CD'].astype(str)
            
            print(f"매핑 전 - 행정구역 수: {len(gdf_merge_2)}, 이동 데이터 수: {len(df_g_h_g_copy)}")
            
            # 이동 데이터와 행정구역 merge
            gdf_merged_3 = gdf_merge_2.merge(df_g_h_g_copy, left_on='ADMI_CD', right_on='O_ADMI_CD', how='left')
            
            print(f"매핑 후 컬럼: {gdf_merged_3.columns.tolist()}")
            print(f"매핑 후 - 총 행정구역: {len(gdf_merged_3)}")
            
            # MOVE_TIME 컬럼 체크
            if 'MOVE_TIME' in gdf_merged_3.columns:
                data_count = len(gdf_merged_3.dropna(subset=['MOVE_TIME']))
                print(f"이동 데이터 있는 지역: {data_count}")
            else:
                print(f"MOVE_TIME 컬럼이 없습니다. 사용 가능한 컬럼: {gdf_merged_3.columns.tolist()}")
                # 기본 지도만 생성
                m = folium.Map(location=[37.5665, 126.9780], zoom_start=11)
                folium.GeoJson(
                    gdf_merge_2,
                    style_function=lambda feature: {
                        'fillColor': '#eeeeee',
                        'color': 'black',
                        'weight': 0.5,
                        'fillOpacity': 0.3,
                        'opacity': 0.8
                    }
                ).add_to(m)
                m.save(output_path)
                return m
            
        else:
            print("ADMI 데이터 로드 실패, 기본 지도 생성")
            gdf_merged_3 = gdf_merge_
        
    except Exception as e:
        print(f"GeoJSON 데이터 로드 실패: {e}")
        import traceback
        traceback.print_exc()
        # 실패 시 기본 지도 생성
        m = folium.Map(location=[37.5665, 126.9780], zoom_start=11)
        m.save(output_path)
        return m
    
    # 지도 생성
    m = folium.Map(
        location=[37.5665, 126.9780], 
        zoom_start=11,
        tiles='OpenStreetMap'
    )
    
    # 이동 데이터가 있는 지역 필터링 (MOVE_TIME 컬럼이 있을 때만)
    if 'MOVE_TIME' in gdf_merged_3.columns:
        # 이동 데이터가 있는 지역
        data_regions = gdf_merged_3.dropna(subset=['MOVE_TIME']).copy()
        
        if len(data_regions) > 0:
            # 로그 스케일 변환
            data_regions['MOVE_TIME_LOG'] = np.log1p(data_regions['MOVE_TIME'])
            
            min_t = data_regions['MOVE_TIME_LOG'].min()
            max_t = data_regions['MOVE_TIME_LOG'].max()
            
            print(f"이동시간 범위: {min_t:.2f} ~ {max_t:.2f} (로그스케일)")
            
            # 색상 맵 생성
            colormap = cm.linear.YlOrRd_09.scale(min_t, max_t)
            colormap.caption = '이동시간 (로그스케일)'
            colormap.add_to(m)
            
            # 데이터 있는 지역 choropleth
            folium.GeoJson(
                data_regions,
                style_function=lambda feature: {
                    'fillColor': colormap(feature['properties']['MOVE_TIME_LOG']),
                    'color': 'black',
                    'weight': 0.5,
                    'fillOpacity': 0.8,
                    'opacity': 1.0
                },
                popup=folium.features.GeoJsonPopup(
                    fields=['FULL_NM', 'MOVE_TIME', 'TOTAL_CNT', 'MOVE_DIST'],
                    aliases=['행정동명', '이동시간(분)', '이동인구수', '이동거리(m)'],
                    localize=True,
                    labels=True
                ),
                tooltip=folium.features.GeoJsonTooltip(
                    fields=['FULL_NM', 'MOVE_TIME', 'TOTAL_CNT'],
                    aliases=['행정동명', '이동시간(분)', '이동인구수'],
                    localize=True
                )
            ).add_to(m)
            
            print(f"Choropleth 지도 생성 완료: {len(data_regions)}개 지역")
        else:
            print("이동 데이터가 있는 지역이 없습니다.")
        
        # 이동 데이터가 없는 지역도 표시 (회색)
        no_data_regions = gdf_merged_3[gdf_merged_3['MOVE_TIME'].isna() | (gdf_merged_3['MOVE_TIME'] == 0)].copy()
        if len(no_data_regions) > 0:
            folium.GeoJson(
                no_data_regions,
                style_function=lambda feature: {
                    'fillColor': '#eeeeee',
                    'color': 'black',
                    'weight': 0.5,
                    'fillOpacity': 0.3,
                    'opacity': 0.8
                },
                tooltip=folium.features.GeoJsonTooltip(
                    fields=['FULL_NM'],
                    aliases=['행정동명'],
                    localize=True
                )
            ).add_to(m)
    else:
        # MOVE_TIME 컬럼이 없으면 모든 지역을 회색으로 표시
        folium.GeoJson(
            gdf_merged_3,
            style_function=lambda feature: {
                'fillColor': '#eeeeee',
                'color': 'black',
                'weight': 0.5,
                'fillOpacity': 0.3,
                'opacity': 0.8
            }
        ).add_to(m)
    
    # 지도 저장
    m.save(output_path)
    print(f"지도 저장 완료: {output_path}")
    
    return m

def create_monthly_maps_and_stats(df_cell_data, year_months, output_dir):
    """월별 지도 및 통계 생성"""
    monthly_maps = {}
    monthly_stats = []
    
    for year, month in year_months:
        print(f"\n{year}-{month:02d} 지도 및 통계 생성 중...")
        
        # 해당 월 데이터 사용 (전체 기간 집계된 데이터)
        df_month = df_cell_data.copy()
        
        # 행정동별 집계
        df_admin_month = df_month.groupby('O_ADMI_CD').agg({
            'MOVE_DIST': 'sum',
            'MOVE_TIME': 'sum',
            'TOTAL_CNT': 'sum',
            'O_CELL_X': 'mean',
            'O_CELL_Y': 'mean'
        }).reset_index()
        
        # 지도 생성
        map_file = os.path.join(output_dir, f"map_{year}_{month:02d}.html")
        create_geojson_visualization(df_admin_month, year, month, map_file)
        monthly_maps[f"{year}-{month:02d}"] = map_file
        
        # 월별 통계 계산 - 직관적 지표로 변경
        if len(df_admin_month) > 0:
            days_in_month = 30  # 간단히 30일로 가정
            
            # 전체 데이터 기준 계산
            total_population = df_admin_month['TOTAL_CNT'].sum()
            total_time = df_admin_month['MOVE_TIME'].sum()
            total_distance = df_admin_month['MOVE_DIST'].sum()
            
            # 직관적 지표 계산
            # 1인 1일 평균 이동시간
            avg_time_per_person_per_day = (total_time / total_population / days_in_month) if total_population > 0 else 0
            
            # 일평균 이동인구수
            daily_avg_population = total_population / days_in_month
            
            # 1인 1일 평균 이동거리
            avg_distance_per_person_per_day = (total_distance / total_population / days_in_month) if total_population > 0 else 0
            
            monthly_stats.append({
                'year_month': f"{year}-{month:02d}",
                'daily_avg_population': daily_avg_population,
                'avg_time_per_person_per_day': avg_time_per_person_per_day,
                'avg_distance_per_person_per_day': avg_distance_per_person_per_day,
                'total_cells': len(df_month),
                'admin_regions': len(df_admin_month)
            })
        else:
            monthly_stats.append({
                'year_month': f"{year}-{month:02d}",
                'daily_avg_population': 0,
                'avg_time_per_person_per_day': 0,
                'avg_distance_per_person_per_day': 0,
                'total_cells': 0,
                'admin_regions': 0
            })
    
    return monthly_maps, monthly_stats


def create_dashboard_html(monthly_maps, monthly_stats, performance_results, year_months, output_path):
    """통합 대시보드 HTML 생성"""
    
    # 기본 정보
    start_date = f"{year_months[0][0]}-{year_months[0][1]:02d}"
    end_date = f"{year_months[-1][0]}-{year_months[-1][1]:02d}"
    analysis_period = f"{start_date} ~ {end_date} ({len(year_months)}개월)"
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>수도권 시민의 출근 시간 분석</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
                padding: 30px;
            }}
            .header {{
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border-radius: 10px;
            }}
            .header h1 {{
                margin: 0;
                font-size: 2.5em;
            }}
            .overview {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .overview h2 {{
                color: #333;
                border-left: 4px solid #667eea;
                padding-left: 15px;
            }}
            .info-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .info-item {{
                background: white;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #ddd;
                text-align: center;
            }}
            .info-item strong {{
                display: block;
                color: #667eea;
                font-size: 1.1em;
            }}
            .maps-section {{
                margin: 30px 0;
            }}
            .maps-section h2 {{
                color: #333;
                border-left: 4px solid #764ba2;
                padding-left: 15px;
            }}
            .map-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .map-item {{
                border: 2px solid #ddd;
                border-radius: 10px;
                overflow: hidden;
                background: white;
            }}
            .map-header {{
                background-color: #667eea;
                color: white;
                padding: 10px;
                text-align: center;
                font-weight: bold;
            }}
            .map-frame {{
                width: 100%;
                height: 400px;
                border: none;
            }}
            .stats-section {{
                margin: 30px 0;
            }}
            .stats-section h2 {{
                color: #333;
                border-left: 4px solid #28a745;
                padding-left: 15px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                background: white;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }}
            th, td {{
                padding: 12px;
                text-align: center;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #28a745;
                color: white;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f8f9fa;
            }}
            tr:hover {{
                background-color: #e8f5e8;
            }}
            .footer {{
                text-align: center;
                margin-top: 30px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 10px;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>수도권 시민의 출근 시간 분석</h1>
                <p>행정동별 Choropleth 지도 기반 시각화</p>
            </div>

            <div class="overview">
                <h2>분석 개요</h2>
                <div class="info-grid">
                    <div class="info-item">
                        <strong>분석 기간</strong>
                        {analysis_period}
                    </div>
                    <div class="info-item">
                        <strong>처리 시간</strong>
                        {performance_results['total_time']:.1f}초 ({performance_results['total_time']/60:.1f}분)
                    </div>
                    <div class="info-item">
                        <strong>메모리 사용량</strong>
                        {performance_results['memory_usage_gb']:.2f} GB
                    </div>
                    <div class="info-item">
                        <strong>생성 지도 수</strong>
                        {len(monthly_maps)}개
                    </div>
                </div>
            </div>

            <div class="maps-section">
                <h2>월별 이동 패턴 지도</h2>
                <p>각 지도는 해당 월의 행정동별 이동 패턴을 choropleth 방식으로 보여줍니다. 색상의 진함 정도는 이동시간을 나타냅니다.</p>
                <div class="map-grid">
    """
    
    # 월별 지도 추가
    for year_month, map_file in monthly_maps.items():
        map_filename = os.path.basename(map_file)
        html_content += f"""
                    <div class="map-item">
                        <div class="map-header">{year_month}</div>
                        <iframe src="{map_filename}" class="map-frame"></iframe>
                    </div>
        """
    
    
    html_content += f"""
                    </tbody>
                </table>
            </div>

            <div class="footer">
                <p>데이터 출처: 수도권 생활이동 데이터</p>
                <p>시각화 방식: 행정동별 Choropleth 맵핑</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # HTML 파일 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"대시보드 HTML 생성 완료: {output_path}")

def run_analysis(period_type):
    """
    기간별 분석 실행
    period_type: '1month', '2months', '3months', '6months', '1year'
    """
    
    # 기간별 년월 설정
    period_configs = {
        '1month': [(2024, 6)],
        '2months': [(2024, 6), (2024, 7)],
        '3months': [(2024, 6), (2024, 7), (2024, 8)],
        '6months': [(2024, 6), (2024, 7), (2024, 8), (2024, 9), (2024, 10), (2024, 11)],
        '1year': [(2024, 6), (2024, 7), (2024, 8), (2024, 9), (2024, 10), (2024, 11),
                  (2024, 12), (2025, 1), (2025, 2), (2025, 3), (2025, 4), (2025, 5)]
    }
    
    if period_type not in period_configs:
        print(f"지원되지 않는 기간 타입: {period_type}")
        print(f"지원 가능한 타입: {list(period_configs.keys())}")
        return
    
    year_months = period_configs[period_type]
    
    # 출력 디렉토리 설정
    output_dir_ = f"/home1/rldnjs16/transit_seoul/results/{period_type}"
    output_dir = f"/home1/rldnjs16/transit_seoul/results/{period_type}/{filename_time }"
    os.makedirs(output_dir_, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"{period_type} 분석 시작")
    print(f"분석 대상: {year_months}")
    print(f"출력 디렉토리: {output_dir}")
    print(f"{'='*60}")
    
    try:
        # 1. 데이터 로드 및 집계
        print("1단계: 데이터 로드 및 집계")
        df_aggregated, performance_results = load_and_aggregate_batch(
            year_months, 
            batch_size=2,
            save_intermediate=True
        )
        
        if df_aggregated is None:
            raise Exception("데이터 로드 실패")
        
        print(f"집계 완료: {len(df_aggregated):,}행")
        
        # 2. 월별 지도 및 통계 생성
        print("2단계: 월별 지도 및 통계 생성")
        monthly_maps, monthly_stats = create_monthly_maps_and_stats(df_aggregated, year_months, output_dir)
        
        # 3. 대시보드 HTML 생성
        print("3단계: 대시보드 HTML 생성")
        dashboard_path = os.path.join(output_dir, "index.html")
        create_dashboard_html(monthly_maps, monthly_stats, performance_results, year_months, dashboard_path)
        
        # 4. 결과 요약
        print(f"\n{'='*60}")
        print("분석 완료 요약")
        print(f"{'='*60}")
        print(f"분석 기간: {period_type}")
        print(f"처리된 월 수: {len(year_months)}")
        print(f"총 격자 수: {len(df_aggregated):,}")
        print(f"총 이동인구: {df_aggregated['TOTAL_CNT'].sum():,}명")
        print(f"평균 이동시간: {df_aggregated['MOVE_TIME'].mean():.1f}분")
        print(f"총 이동거리: {df_aggregated['MOVE_DIST'].sum():,.0f}m")
        print(f"처리 시간: {performance_results['total_time']:.1f}초")
        print(f"메모리 사용량: {performance_results['memory_usage_gb']:.2f} GB")
        print(f"대시보드: {dashboard_path}")
        print(f"{'='*60}")
        
        return True
        
    except Exception as e:
        print(f"분석 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 메모리 사용량 확인
    print("현재 시스템 상태:")
    monitor_memory_usage()
    print()
    
    choice = '5'
    
    period_map = {
        '1': '1month',
        '2': '2months', 
        '3': '3months',
        '4': '6months',
        '5': '1year'
    }
    
    if choice in period_map:
        # 선택된 기간 분석 실행
        success = run_analysis(period_map[choice])
        if success:
            print(f"\n{period_map[choice]} 분석 성공")
        else:
            print(f"\n{period_map[choice]} 분석 오류")
        
    else:
        print("오류")
    
    # 최종 메모리 상태 확인
    print("\n최종 메모리 상태:")
    monitor_memory_usage()