#%%
import duckdb
import pandas as pd
import gc

#%%
# DuckDB 설정
duckdb.sql("PRAGMA threads=2")
duckdb.sql("PRAGMA memory_limit='512MB'")

#%%
# 처리할 년월 리스트
year_months = [(2024, 6), (2024, 7), (2024, 8), (2024, 9), (2024, 10), (2024, 11)]
batch_size = 2

# 배치별 처리
for I in range(0, len(year_months), batch_size):
    batch = year_months[I:I+batch_size]
    batch_data = []  # 배치 내 데이터 수집용 리스트
    
    # 배치 내 각 월 데이터 로딩
    for year, month in batch:
        query = f"""
            SELECT 
                O_ADMI_CD, O_CELL_ID, O_CELL_X, O_CELL_Y,
                MOVE_DIST, MOVE_TIME, TOTAL_CNT
            FROM '/home1/rldnjs16/transit_seoul/dataset/data_month/year={year:04}/month={month:02}/SEOUL_PURPOSE_250M_IN_{year:04}{month:02}.parquet'
            WHERE MOVE_PURPOSE = 1
        """
        
        # 데이터 로딩
        table = duckdb.query(query).arrow()
        df = table.to_pandas(split_blocks=True, self_destruct=True)
        
        # 즉시 집계하여 메모리 절약
        df_agg = df.groupby(['O_ADMI_CD', 'O_CELL_ID']).agg({
            'MOVE_DIST': 'sum',
            'MOVE_TIME': 'sum',
            'TOTAL_CNT': 'sum',
            'O_CELL_X': 'first',
            'O_CELL_Y': 'first'
        }).reset_index()
        
        # 배치 데이터에 추가
        batch_data.append(df_agg)
        
        # 메모리 정리
        del df, table
        gc.collect()

# 배치 내 데이터 결합 및 재집계
batch_combined = pd.concat(batch_data, ignore_index=True)
del batch_data
