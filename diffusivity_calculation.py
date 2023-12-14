import numpy as np
import cv2

def calculate_diffusivity(image):
    # Пример: Анализ градиентов интенсивности
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Вычисление среднего градиента как меры диффузии
    diffusivity = np.mean(gradient_magnitude)

    return diffusivity


def fitzhugh_nagumo_diffusion(image, num_iterations=100, alpha=0.1, beta=0.2):
    """
    Рассчитывает диффузию на SEM-изображении с использованием алгоритма ФицХью-Нагумо.

    Parameters:
    - image: SEM-изображение в формате NumPy array.
    - num_iterations: количество итераций алгоритма.
    - alpha: параметр альфа для алгоритма ФицХью-Нагумо.
    - beta: параметр бета для алгоритма ФицХью-Нагумо.

    Returns:
    - diffused_image: SEM-изображение после применения диффузии.
    """
    diffused_image = image.copy().astype(np.float32)

    for _ in range(num_iterations):
        laplacian = cv2.Laplacian(diffused_image, cv2.CV_64F)
        diffusion_term = alpha * laplacian - beta * (diffused_image - image)
        diffused_image += diffusion_term

    # Ограничение значений изображения в пределах 0-255
    diffused_image = np.clip(diffused_image, 0, 255).astype(np.uint8)

    return diffused_image

# Пример использования на SEM-изображении
image = cv2.imread('your_sem_image.jpg', cv2.IMREAD_GRAYSCALE)

diffusivity_value = calculate_diffusivity(image)
print(f'Diffusivity: {diffusivity_value:.2f}')

# Вычислить диффузию
diffused_result = fitzhugh_nagumo_diffusion(image)

# Вывести результат
cv2.imshow('Original SEM Image', image)
cv2.imshow('Diffused SEM Image', diffused_result)
cv2.waitKey(0)
cv2.destroyAllWindows()
