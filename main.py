import numpy as np
from PIL import Image, ImageFont, ImageDraw

FONTS_PATH = 'fonts/'
NUM_EPOCHS = 10000  # Количество эпох обучения
LEARNING_RATE = 1  # Скорость обучения
WEIGHT_RANGE = (-0.00003, 0.00003)  # Диапазон для инициализации весов

class Perceptron:
    def __init__(self, num_inputs, num_classes):
        self.weights = np.random.uniform(*WEIGHT_RANGE, (num_classes, num_inputs))
        self.bias = np.zeros(num_classes)
        self.learning_rate = LEARNING_RATE
        self.num_classes = num_classes

    def activate(self, inputs):
        """Функция активации персептрона"""
        net_input = np.dot(self.weights, inputs) + self.bias
        return np.argmax(net_input)

    def train(self, X, y, epochs):
        """Обучение персептрона методом коррекции по ошибке через дельта-правило"""
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in zip(X, y):
                prediction = self.activate(inputs)
                error = target - prediction
                self.weights[prediction] += self.learning_rate * error * inputs
                self.bias[prediction] += self.learning_rate * error
                total_error += np.abs(error)
            print(f'Эпоха {epoch+1}/{epochs}, Общая Ошибка: {total_error:.2f}')

    def predict(self, X):
        """Получение выходов персептрона для новых входных данных"""
        return [self.activate(inputs) for inputs in X]


def main():
    # Определение символов для распознавания
    symbols = ['A', 'B', 'C', 'D']
    num_classes = len(symbols)

    # Загрузка и сохранение обучающих образов
    X_train = []
    y_train = []
    for font_num in range(1, 5):
        for symbol in symbols:
            font = ImageFont.truetype(f'{FONTS_PATH}font{font_num}.ttf', size=64)
            image = Image.new('L', (64, 64), 0)
            draw = ImageDraw.Draw(image)
            draw.text((0, 0), symbol, font=font, fill=255)
            image.save(f'train_{symbol}_{font_num}.png')
            X_train.append(np.array(image).flatten())
            y_train.append(symbols.index(symbol))

    X_train = np.array(X_train) / 255.0
    y_train = np.array(y_train)

    # Создание и обучение модели
    perceptron = Perceptron(num_inputs=64 * 64, num_classes=num_classes)
    perceptron.train(X_train, y_train, epochs=NUM_EPOCHS)

    # Загрузка и сохранение тестовых образов
    X_test = []
    y_test = []
    font = ImageFont.truetype(f'{FONTS_PATH}font5.ttf', size=64)
    for symbol in symbols:
        image = Image.new('L', (64, 64), 0)
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), symbol, font=font, fill=255)
        image.save(f'test_{symbol}.png')
        X_test.append(np.array(image).flatten())
        y_test.append(symbols.index(symbol))

    X_test = np.array(X_test) / 255.0
    y_test = np.array(y_test)

    # Оценка модели на тестовой выборке
    predictions = perceptron.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f'Точность: {accuracy * 100:.2f}%')

    # Тест на рандомном изображении
    random_index = np.random.randint(0, len(X_test))
    random_image = X_test[random_index]
    random_prediction = perceptron.activate(random_image)
    print(f'Случайное изображение предсказано как: {symbols[random_prediction]}')

if __name__ == '__main__':
    main()