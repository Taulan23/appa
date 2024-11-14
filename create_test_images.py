import numpy as np
import cv2
from typing import Tuple
import logging
from pathlib import Path

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestImageGenerator:
    def __init__(self, size: Tuple[int, int] = (512, 512)):
        """
        Инициализация генератора тестовых изображений
        
        Args:
            size: Размер генерируемых изображений (ширина, высота)
        """
        self.size = size
        self.output_dir = Path('test_images')
        self.output_dir.mkdir(exist_ok=True)

    def create_high_frequency_pattern(self) -> np.ndarray:
        """
        Создает изображение с высокочастотными паттернами
        """
        img = np.ones(self.size, dtype=np.uint8) * 255
        
        # Добавляем сетку
        for i in range(0, self.size[0], 20):
            cv2.line(img, (i, 0), (i, self.size[1]), (0, 0, 0), 1)
            cv2.line(img, (0, i), (self.size[0], i), (0, 0, 0), 1)
        
        # Добавляем текст разных размеров
        fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_DUPLEX]
        texts = ['High', 'Frequency', 'Pattern']
        
        for i, (font, text) in enumerate(zip(fonts, texts)):
            cv2.putText(img, text, (50, 100 + i * 100), font, 1.5, (0, 0, 0), 2)
        
        # Добавляем круги разных размеров
        for i in range(5):
            radius = 10 + i * 20
            cv2.circle(img, (400, 400), radius, (0, 0, 0), 1)
        
        return img

    def create_synthetic_scene(self) -> np.ndarray:
        """
        Создает синтетическую сцену с разными геометрическими формами
        """
        img = np.zeros((*self.size, 3), dtype=np.uint8)
        
        # Градиентный фон
        for i in range(self.size[0]):
            color = int((i / self.size[0]) * 255)
            img[:, i] = [color, color, color]
        
        # Добавляем геометрические фигуры
        cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), -1)
        cv2.circle(img, (350, 250), 80, (0, 0, 255), -1)
        
        # Добавляем треугольник
        pts = np.array([[250, 350], [350, 450], [150, 450]], np.int32)
        cv2.fillPoly(img, [pts], (255, 0, 0))
        
        return img

    def create_frequency_test_pattern(self) -> np.ndarray:
        """
        Создает тестовый паттерн с различными частотами
        """
        img = np.ones(self.size, dtype=np.uint8) * 255
        
        # Создаем синусоидальные паттерны разной частоты
        x = np.linspace(0, 20*np.pi, self.size[0])
        for i, freq in enumerate([1, 2, 4, 8, 16]):
            y = np.sin(freq * x)
            y = (y + 1) * 100 + i * 100
            pts = np.array([[j, int(y[j])] for j in range(self.size[0])])
            cv2.polylines(img, [pts.reshape((-1, 1, 2))], False, (0, 0, 0), 2)
        
        return img

    def create_noise_pattern(self) -> np.ndarray:
        """
        Создает изображение с разными типами шума
        """
        img = np.ones((*self.size, 3), dtype=np.uint8) * 255
        
        # Гауссов шум
        noise = np.random.normal(0, 50, (*self.size, 3))
        img_gaussian = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        # Соль и перец
        prob = 0.05
        noise_mask = np.random.random(self.size[:2])
        img_sp = img.copy()
        img_sp[noise_mask < prob/2] = 0
        img_sp[noise_mask > 1 - prob/2] = 255
        
        # Комбинируем разные типы шума в одном изображении
        img[:, :self.size[1]//2] = img_gaussian[:, :self.size[1]//2]
        img[:, self.size[1]//2:] = img_sp[:, self.size[1]//2:]
        
        return img

    def generate_all_images(self):
        """
        Генерирует и сохраняет все тестовые изображения
        """
        # Словарь с функциями генерации и именами файлов
        generators = {
            'high_frequency.jpg': self.create_high_frequency_pattern,
            'synthetic_scene.jpg': self.create_synthetic_scene,
            'frequency_pattern.jpg': self.create_frequency_test_pattern,
            'noise_pattern.jpg': self.create_noise_pattern
        }
        
        for filename, generator in generators.items():
            try:
                logger.info(f"Генерация изображения: {filename}")
                img = generator()
                output_path = self.output_dir / filename
                cv2.imwrite(str(output_path), img)
                logger.info(f"Изображение сохранено: {output_path}")
            except Exception as e:
                logger.error(f"Ошибка при создании {filename}: {str(e)}")

def main():
    # Создаем генератор с размером изображения 512x512
    generator = TestImageGenerator((512, 512))
    
    # Генерируем все тестовые изображения
    generator.generate_all_images()

if __name__ == "__main__":
    main()