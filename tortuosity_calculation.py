def calculate_tortuosity(image):
    # Пример: Извлечение структурных особенностей
    gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Вычисление средней кривизны или извилистости
    tortuosity = np.mean(gradient_magnitude)

    return tortuosity

# Пример использования
tortuosity_value = calculate_tortuosity(processed_img)
print(f'Tortuosity: {tortuosity_value:.2f}')
