import cv2
import numpy as np
from skimage import feature

def preprocess_image(image_path):
    # Загрузка SEM-изображения
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    return img

def calculate_strength(image):
    # Извлечение текстурных особенностей с использованием метода градиентного выделения
    edges = feature.canny(image, sigma=2)

    # Вычисление структурных параметров, например, плотности контуров
    contour_density = np.sum(edges) / (image.shape[0] * image.shape[1])

    # Моделирование прочности на основе структурных параметров
    strength = contour_density * 100  # Пример моделирования, может потребоваться коррекция

    return strength

# Пример использования
image_path = 'path/to/sem_image.jpg'

# Обработка изображения
processed_img = preprocess_image(image_path)

# Вычисление прочности
strength_value = calculate_strength(processed_img)

# Вывод результатов
print(f'Strength: {strength_value:.2f}')
