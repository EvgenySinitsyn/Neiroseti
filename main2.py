import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Определение трансформаций для данных
image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Загрузка данных
mnist_train_data = datasets.MNIST(root='./data', train=True, download=True, transform=image_transforms)
mnist_test_data = datasets.MNIST(root='./data', train=False, download=True, transform=image_transforms)

# Создание DataLoader (Шаг 1)
training_data_loader = DataLoader(mnist_train_data, batch_size=64, shuffle=True)
testing_data_loader = DataLoader(mnist_test_data, batch_size=64, shuffle=False)

# Определение модели
class SimplePerceptronModel(nn.Module):
    def __init__(self):
        super(SimplePerceptronModel, self).__init__()
        # Шаг 2: Инициализация весов и смещений
        self.layer = nn.Linear(28 * 28, 10)

    def forward(self, input_images):
        input_images = input_images.view(-1, 28 * 28)
        output = self.layer(input_images)
        return output

model_instance = SimplePerceptronModel()

# Определение функции потерь и оптимизатора
loss_function = nn.CrossEntropyLoss()
optimizer_for_model = optim.SGD(model_instance.parameters(), lr=0.01, momentum=0.9)

# Количество эпох и порог ошибки для остановки
number_of_training_epochs = 10
threshold_for_stopping = 0.001

# Обучение модели
for current_epoch in range(number_of_training_epochs):
    accumulated_loss = 0.0
    # Шаг 3: Случайный выбор пары (X, D)m из обучающей выборки
    for index, data_batch in enumerate(training_data_loader, 0):
        input_images, target_labels = data_batch
        optimizer_for_model.zero_grad() # Шаг 8: Коррекция синаптических весов и нейронных смещений

        # Шаг 4: Прямое распространение и вычисление ошибки
        output_predictions = model_instance(input_images)

        # Шаг 6: Вычисление ошибки ε для текущего обучающего вектора
        loss_value = loss_function(output_predictions, target_labels)
        loss_value.backward()

        # Здесь происходит коррекция весов модели
        optimizer_for_model.step()

        accumulated_loss += loss_value.item()
        if index % 200 == 199:
            print(f'[Эпоха: {current_epoch + 1}, Итерация: {index + 1}] Потеря: {accumulated_loss / 200:.3f}')
            accumulated_loss = 0.0

    # Шаг 7: Проверка критерия останова
    average_loss = accumulated_loss / len(training_data_loader)
    if average_loss < threshold_for_stopping:
        print("Ранняя остановка из-за низкой средней потери")
        break

# Тестирование модели
accuracy_score = 0
total_samples = 0
predictions_list = []
actual_labels_list = []

with torch.no_grad():
    for data_sample in testing_data_loader:
        image_samples, label_samples = data_sample
        prediction_results = model_instance(image_samples)
        _, predicted_labels = torch.max(prediction_results.data, 1)
        predictions_list.extend(predicted_labels.tolist())
        actual_labels_list.extend(label_samples.tolist())

total_samples += len(predictions_list)
accuracy_score += sum([p == l for p, l in zip(predictions_list, actual_labels_list)])

print(f'Точность сети на тестовых изображениях: {100 * accuracy_score / total_samples:.2f}%')

# Вывод фактических и угаданных символов
print("Угаданные символы:", predictions_list[:20])
print("Фактические метки:", actual_labels_list[:20])
