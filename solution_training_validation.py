import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from skimage import filters
from skimage import exposure
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

from porosity_calculation import calculate_porosity
from permeability_calculation import calculate_permeability
from diffusivity_calculation import calculate_diffusivity
from tortuosity_calculation import calculate_tortuosity

# Пример SemDataset для задачи сегментации пор SEM-изображений
class SemDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        Конструктор класса SemDataset.

        Parameters:
        - root: Путь к корневой директории датасета.
        - transform: Трансформации для применения к изображениям.
        """
        # self.dataset = SemDataset(root, download=True)
        self.transform = transform

    def __len__(self):
        """
        Возвращает общее количество элементов в датасете.

        Returns:
        - len: Общее количество элементов.
        """
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        Возвращает изображение и метку (target) для сегментации.

        Parameters:
        - idx: Индекс элемента.

        Returns:
        - image: Тензор с изображением.
        - target: Тензор с меткой для сегментации.
        """
        image, target = self.dataset[idx]

        # Применяем трансформации, если они заданы
        if self.transform:
            image, target = self.transform(image, target)

        # Возвращаем изображение и метку (target) для сегментации
        return image, target

# Функция для выделения краев и создания маски с использованием адаптивной бинаризации
def preprocess_image(image, mask):
    """
    Преобразует изображение и маску для сегментации.

    Parameters:
    - image: Изображение.
    - mask: Маска для сегментации.

    Returns:
    - edges: Тензор с выделенными краями.
    - binary_mask: Тензор с созданной маской после адаптивной бинаризации.
    """
    # Применяем оператор Собеля для выделения краев
    edges = filters.sobel(image)

    # Применяем адаптивную бинаризацию для создания маски
    threshold = filters.threshold_adaptive(edges, block_size=101)
    binary_mask = edges > threshold

    return torch.from_numpy(edges).float(), torch.from_numpy(binary_mask).float()

def train_model(input_path, output_path):
    # Пример transform с предобработкой
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Пример трансформации, добавьте свои
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.expand(3, -1, -1)),  # Приводим к трем каналам
        transforms.Lambda(lambda x: exposure.equalize_adapthist(x.numpy(), clip_limit=0.03)),
        transforms.Lambda(lambda x: preprocess_image(x[0], x[1])),
    ])

    # Пример train_loader
    train_dataset = SemDataset(root='./data', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

    # Обучение модели
    num_epochs = 40
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}')

    # Сохранение обученной модели
    torch.save(model.state_dict(), 'segmentation_model.pth')

def calculate_dice_coefficient(ground_truth_mask, predicted_mask):
    # Перевести маски в тип данных bool
    ground_truth_mask = ground_truth_mask.astype(bool)
    predicted_mask = predicted_mask.astype(bool)

    # Вычислить площадь для каждой маски
    area_ground_truth = np.sum(ground_truth_mask)
    area_predicted = np.sum(predicted_mask)

    # Вычислить пересечение масок
    intersection = np.sum(np.logical_and(ground_truth_mask, predicted_mask))

    # Вычислить DICE-коэффициент
    dice_coefficient = (2.0 * intersection) / (area_ground_truth + area_predicted)

    return dice_coefficient


def evaluate_model(model, test_loader, device):
    """
    Оценка производительности модели на тестовом наборе данных.

    Parameters:
    - model: Обученная модель.
    - test_loader: DataLoader для тестового набора данных.
    - device: Устройство (cuda или cpu).

    Returns:
    - accuracy: Точность модели на тестовом наборе данных.
    - fpr, tpr, roc_auc: Данные для построения ROC-кривой.
    """
    model.eval()  # Устанавливаем модель в режим оценки
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            predictions = torch.sigmoid(outputs) > 0.5

            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    accuracy = accuracy_score(all_labels.flatten(), all_predictions.flatten())

    # Рассчитываем ROC-кривую и площадь под кривой (AUC)
    fpr, tpr, _ = roc_curve(all_labels.flatten(), all_predictions.flatten())
    roc_auc = auc(fpr, tpr)

    # Рассчитываем DICE коэффициент
    dice_coefficient = calculate_dice_coefficient(all_labels.numpy(), all_predictions.numpy())

    # Вычисляем параметры материала
    porosity = calculate_porosity(all_predictions.numpy())
    permeability = calculate_permeability(all_predictions.numpy())
    diffusivity = calculate_diffusivity(all_predictions.numpy())
    tortuosity = calculate_tortuosity(all_predictions.numpy())

    return accuracy, fpr, tpr, roc_auc, dice_coefficient, porosity, permeability, diffusivity, tortuosity

def predict_with_material_properties(model, image, device):
    """
    Предсказание маски и вычисление параметров материала на изображении.

    Parameters:
    - model: Обученная модель.
    - image: Тензор с изображением.
    - device: Устройство (cuda или cpu).

    Returns:
    - predicted_mask: Тензор с предсказанной маской.
    - porosity: Значение пористости материала.
    - permeability: Значение проницаемости материала.
    - diffusivity: Значение диффузии материала.
    - tortuosity: Значение тортуозности материала.
    """
    model.eval()
    with torch.no_grad():
        image = image.to(device).unsqueeze(0)  # Добавляем размерность батча
        output = model(image)
        segmentation_probabilities = torch.sigmoid(output)
        predicted_mask = segmentation_probabilities > 0.5

    # Вычисляем параметры материала
    porosity = calculate_porosity(predicted_mask.cpu().numpy())
    permeability = calculate_permeability(predicted_mask.cpu().numpy())
    diffusivity = calculate_diffusivity(predicted_mask.cpu().numpy())
    tortuosity = calculate_tortuosity(predicted_mask.cpu().numpy())

    return predicted_mask.float().cpu().numpy(), porosity, permeability, diffusivity, tortuosity



# Пример использования evaluate_model
test_dataset = SemDataset(root='./test_data', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

model = SegmentationModel().to(device)
model.load_state_dict(torch.load('segmentation_model.pth'))
model.eval()

(test_accuracy, fpr, tpr, roc_auc, dice_coefficient,
 porosity, permeability, diffusivity, tortuosity) = evaluate_model(model, test_loader, device)

print(f'Test Accuracy: {test_accuracy:.4f}')
print(f'Dice Coefficient: {dice_coefficient:.4f}')
print(f'Porosity: {porosity:.4f}')
print(f'Permeability: {permeability:.4f}')
print(f'Diffusivity: {diffusivity:.4f}')
print(f'Tortuosity: {tortuosity:.4f}')

# Демонстрация ROC-кривой и AUC
plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# Пример использования функции predict_with_material_properties
sample_image, _ = next(iter(test_loader))
sample_image = sample_image.to(device)

predicted_mask, porosity, permeability, diffusivity, tortuosity = predict_with_material_properties(model, sample_image, device)

print(f'Porosity: {porosity:.4f}')
print(f'Permeability: {permeability:.4f}')
print(f'Diffusivity: {diffusivity:.4f}')
print(f'Tortuosity: {tortuosity:.4f}')