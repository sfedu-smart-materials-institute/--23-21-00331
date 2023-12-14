import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.utils as utils

# Определение генератора
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Linear(100, 256)
        self.relu = nn.ReLU()
        self.deconv = nn.ConvTranspose2d(256, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = x.view(-1, 256, 1, 1)
        x = self.deconv(x)
        x = torch.sigmoid(x)
        return x

# Определение дискриминатора
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64*14*14, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(-1, 64*14*14)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
    
def predict(generator, latent_size, num_samples):
    """
    Генерация новых изображений с использованием обученного генератора.

    Args:
    - generator (nn.Module): Обученный генератор.
    - latent_size (int): Размерность входного случайного шума.
    - num_samples (int): Количество изображений для генерации.

    Returns:
    - generated_images (torch.Tensor): Тензор сгенерированных изображений.
    """
    generator.eval()
    with torch.no_grad():
        # Генерация изображений
        noise = Variable(torch.randn(num_samples, latent_size))
        generated_images = generator(noise)

    generator.train()
    return generated_images

# Гиперпараметры и инициализация
batch_size = 64
latent_size = 100
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Загрузка данных
dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Инициализация генератора, дискриминатора и оптимизаторов
generator = Generator()
discriminator = Discriminator()

criterion = nn.BCELoss()
optimizer_generator = optim.Adam(generator.parameters(), lr=0.0002)
optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=0.0002)

# Обучение GAN
num_epochs = 100

for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        # Получение данных
        real_data, _ = data
        target_real = Variable(torch.ones(real_data.size(0), 1))
        target_fake = Variable(torch.zeros(real_data.size(0), 1))

        # Обучение дискриминатора на реальных данных
        discriminator.zero_grad()
        output_real = discriminator(real_data)
        loss_real = criterion(output_real, target_real)
        loss_real.backward()

        # Обучение дискриминатора на сгенерированных данных
        noise = Variable(torch.randn(real_data.size(0), latent_size))
        fake_data = generator(noise).detach()
        output_fake = discriminator(fake_data)
        loss_fake = criterion(output_fake, target_fake)
        loss_fake.backward()
        optimizer_discriminator.step()

        # Обучение генератора
        generator.zero_grad()
        output = discriminator(fake_data)
        loss_generator = criterion(output, target_real)
        loss_generator.backward()
        optimizer_generator.step()

        # Вывод промежуточных результатов
        if i % 100 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     (loss_real + loss_fake).item(), loss_generator.item()))

    # Валидация модели после каждой эпохи
    generator.eval()
    with torch.no_grad():
        # Генерация изображений
        validation_noise = Variable(torch.randn(num_validation_samples, latent_size))
        generated_images = generator(validation_noise)

        # Визуализация и сохранение изображений
        generated_images_grid = utils.make_grid(generated_images, nrow=4, normalize=True)
        utils.save_image(generated_images_grid, f'generated_images_epoch_{epoch}.png')

    generator.train()