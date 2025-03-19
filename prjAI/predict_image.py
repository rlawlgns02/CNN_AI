import cv2
from keras.models import load_model
import numpy as np

# 이미지 전처리 함수
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # 이미지 크기 조정
    image = image / 255.0  # 픽셀 값 정규화
    return image

# 모델 불러오기
model = load_model('image_classifier.h5')

# 이미지 경로
image_path = r'C:\Users\503\Desktop\2025_Robot\prjAI\img\A2.jpg'
image = preprocess_image(image_path)

# 배치 형식으로 변경
image = np.expand_dims(image, axis=0)

# 예측
prediction = model.predict(image)

# 예측 결과 해석 및 등급 출력
grades = ['A', 'B', 'C', 'D', 'F']
for i, grade in enumerate(grades):
    print(f"{grade} 등급: {prediction[0][i] * 100:.2f}%")

predicted_grade = grades[np.argmax(prediction)]
print(f"가장 높은 확률의 등급: {predicted_grade}")