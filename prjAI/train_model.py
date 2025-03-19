import os
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 데이터 디렉토리 생성
os.makedirs('train_data/A', exist_ok=True)
os.makedirs('train_data/B', exist_ok=True)
os.makedirs('train_data/C', exist_ok=True)
os.makedirs('train_data/D', exist_ok=True)
os.makedirs('train_data/F', exist_ok=True)

# 이미지 전처리 함수
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # 이미지 크기 조정
    image = image / 255.0  # 픽셀 값 정규화
    return image

# 모델 생성
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='softmax'))  # 5개 등급 분류

# 데이터 생성기
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'train_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# 모델 컴파일 및 학습
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_generator, epochs=10)  # 학습 횟수 설정

# 모델 저장
model.save('image_classifier.h5')