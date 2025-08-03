from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd

# 1. 데이터 로딩 (sklearn 내장)
housing = fetch_california_housing()
X, y = housing.data, housing.target
print(f"데이터 로딩: {len(X):,}개 집")

# 2. 훈련/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("모델 학습 완료")

# 4. 예측 및 평가
predictions = model.predict(X_test)
score = r2_score(y_test, predictions)
print(f"예측 정확도: {score:.3f} ({score*100:.1f}%)")

# 5. 예측 예시
print("\n예측 예시:")
for i in range(3):
    actual = y_test[i]
    pred = predictions[i]
    print(f"실제: ${actual:.1f}만, 예측: ${pred:.1f}만 (오차: ${abs(actual-pred):.1f}만)")

print("분석 완료")
