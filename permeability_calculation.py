import cv2
import numpy as np
from skimage import measure

def calculate_permeability(image_path):
    # Загрузка изображения SEM
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Нормализация яркости в диапазоне [0, 1]
    normalized_image = image.astype(float) / 255.0

    # Применение порогового значения для выделения пор
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)

    # Нахождение контуров
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Измерение геометрических параметров пор с использованием scikit-image
    total_area = 0
    total_perimeter = 0
    total_major_axis = 0
    total_minor_axis = 0

    for contour in contours:
        # Измерение площади, периметра и осей контура
        props = measure.regionprops(contour.astype(int))[0]
        
        total_area += props.area
        total_perimeter += props.perimeter
        total_major_axis += props.major_axis_length
        total_minor_axis += props.minor_axis_length

    # Вычисление средней яркости пикселей
    mean_brightness = np.mean(normalized_image)

    # Вычисление проницаемости на основе геометрических параметров и яркости
    permeability_geometry = total_area / total_perimeter
    permeability_brightness = mean_brightness

    return permeability_geometry, permeability_brightness, total_major_axis, total_minor_axis

# Пример использования
image_path = "path/to/your/image.jpg"
permeability_geometry, permeability_brightness, major_axis, minor_axis = calculate_permeability(image_path)
print("Проницаемость (геометрическая):", permeability_geometry)
print("Проницаемость (на основе яркости):", permeability_brightness)
print("Средняя большая ось:", major_axis)
print("Средняя малая ось:", minor_axis)
