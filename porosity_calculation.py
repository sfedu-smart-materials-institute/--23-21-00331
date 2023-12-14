import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    # Загрузка SEM-изображения
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Применение фильтров
    img = cv2.medianBlur(img, 5)

    return img

def binarize_image(img):
    # Бинаризация изображения
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary_img

def calculate_porosity(binary_img):
    # Анализ структуры
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    total_area = binary_img.shape[0] * binary_img.shape[1]
    pore_area = 0

    for contour in contours:
        pore_area += cv2.contourArea(contour)

    # Вычисление пористости
    porosity = (pore_area / total_area) * 100

    return porosity

def visualize_results(img, binary_img):
    # Визуализация исходного изображения и бинаризированного изображения
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(img, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title('Binary Image')
    plt.imshow(binary_img, cmap='gray')

    plt.show()

# Пример использования
image_path = 'path/to/sem_image.jpg'

# Обработка изображения
processed_img = preprocess_image(image_path)

# Бинаризация изображения
binary_image = binarize_image(processed_img)

# Вычисление пористости
porosity_value = calculate_porosity(binary_image)

# Визуализация результатов
visualize_results(processed_img, binary_image)

# Вывод пористости
print(f'Porosity: {porosity_value:.2f}%')
