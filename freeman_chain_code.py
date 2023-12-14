import numpy as np
import cv2

def freeman_chain_code(contour):
    """
    Вычисляет цепной код Фримена для заданного контура.

    Parameters:
    - contour: список точек контура в формате [(x1, y1), (x2, y2), ...]

    Returns:
    - chain_code: список цепных кодов направлений
    """
    chain_code = []

    # Пройти по каждой точке контура, начиная с первой
    for i in range(1, len(contour)):
        # Вычислить изменение координат между текущей и предыдущей точками
        dx = contour[i][0] - contour[i-1][0]
        dy = contour[i][1] - contour[i-1][1]

        # Определить код направления
        direction = (dx, dy)
        code = get_freeman_code(direction)

        # Добавить код в список цепных кодов
        chain_code.append(code)

    return chain_code

def get_freeman_code(direction):
    """
    Определить цепной код Фримена для заданного направления.

    Parameters:
    - direction: кортеж (dx, dy) с изменением координат

    Returns:
    - code: цепной код направления
    """
    # Коды направлений в соответствии с восьмью направлениями Фримена
    freeman_codes = [3, 2, 1, 4, 0, 0, 5, 6, 7]

    # Получить индекс кода направления
    index = (direction[1] + 1) * 3 + (direction[0] + 1)

    # Получить цепной код
    code = freeman_codes[index]

    return code

# Пример использования на изображении
image = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)
_, binary_image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) > 0:
    # Выберем первый контур (можно выбрать любой другой)
    selected_contour = contours[0]

    # Вычислим цепной код Фримена для выбранного контура
    code_result = freeman_chain_code(selected_contour)

    # Выведем результат
    print("Цепной код Фримена:", code_result)
else:
    print("Контур не найден.")
