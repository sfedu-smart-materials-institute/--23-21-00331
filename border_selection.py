import cv2
import numpy as np

def edge_detection_sem(image_path, threshold_low=30, threshold_high=150):
    """
    Выполняет выделение краев на SEM-изображении с использованием операторов Кэнни и Собеля.

    Parameters:
    - image_path: путь к SEM-изображению.
    - threshold_low: нижний порог для оператора Кэнни.
    - threshold_high: верхний порог для оператора Кэнни.

    Returns:
    - edges_canny: изображение с выделенными краями (оператор Кэнни).
    - edges_sobel: изображение с выделенными краями (оператор Собеля).
    """
    # Загрузка SEM-изображения в оттенках серого
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Применение оператора Собеля для выделения краев
    sobelx = cv2.Sobel(original_image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(original_image, cv2.CV_64F, 0, 1, ksize=5)
    edges_sobel = np.sqrt(sobelx**2 + sobely**2)
    edges_sobel = np.uint8(edges_sobel)

    # Применение оператора Кенни для улучшения выделения границ
    edges_canny = cv2.Canny(np.uint8(sobel_mag), 50, 150)

    # Отображение изображений
    cv2.imshow('Original SEM Image', original_image)
    cv2.imshow('Edges Detection (Canny)', edges_canny)
    cv2.imshow('Edges Detection (Sobel)', edges_sobel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return edges_canny, edges_sobel

# Пример использования
image_path = 'your_sem_image.jpg'
edges_canny_result, edges_sobel_result = edge_detection_sem(image_path)
