import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack
import cv2
from pathlib import Path
import logging
from typing import Tuple, Optional
from glob import glob

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageFilter:
    def __init__(self, cutoff_frequency: float = 0.1):
        """
        Инициализация фильтра изображения
        
        Args:
            cutoff_frequency (float): Частота среза (0 до 1)
        """
        self.cutoff_frequency = cutoff_frequency
        self.output_dir = Path('output')
        self.output_dir.mkdir(exist_ok=True)
        
        # Создаем поддиректории для разных типов выходных файлов
        self.filtered_dir = self.output_dir / 'filtered'
        self.spectrum_dir = self.output_dir / 'spectrum'
        self.analysis_dir = self.output_dir / 'analysis'
        
        for dir_path in [self.filtered_dir, self.spectrum_dir, self.analysis_dir]:
            dir_path.mkdir(exist_ok=True)

    def _create_frequency_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Создает частотную маску для фильтра"""
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        
        y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
        d = np.sqrt(x*x + y*y)
        d_normalized = d / np.sqrt(crow**2 + ccol**2)
        
        return (d_normalized < self.cutoff_frequency).astype(float)

    def apply_filter(self, image: np.ndarray, preserve_color: bool = True) -> np.ndarray:
        """
        Применяет низкочастотный фильтр к изображению
        
        Args:
            image: Входное изображение
            preserve_color: Сохранять ли цвета изображения
            
        Returns:
            np.ndarray: Отфильтрованное изображение
        """
        if preserve_color and len(image.shape) == 3:
            # Обработка каждого цветового канала отдельно
            result = np.zeros_like(image)
            for i in range(3):
                channel = image[:, :, i]
                result[:, :, i] = self._filter_channel(channel)
            return result
        else:
            # Обработка в градациях серого
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return self._filter_channel(image)

    def _filter_channel(self, channel: np.ndarray) -> np.ndarray:
        """Применяет фильтр к одному каналу изображения"""
        f_transform = fftpack.fft2(channel.astype(float))
        f_shift = fftpack.fftshift(f_transform)
        
        mask = self._create_frequency_mask(channel.shape)
        f_filtered = f_shift * mask
        
        f_ishift = fftpack.ifftshift(f_filtered)
        img_filtered = fftpack.ifft2(f_ishift)
        img_filtered = np.abs(img_filtered)
        
        # Нормализация
        img_filtered = ((img_filtered - np.min(img_filtered)) / 
                       (np.max(img_filtered) - np.min(img_filtered)) * 255)
        
        return img_filtered.astype(np.uint8)

    def visualize_spectrum(self, f_shift: np.ndarray) -> np.ndarray:
        """Визуализирует спектр частот"""
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)
        return cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)

    def process_image(self, image_path: str, save_results: bool = True) -> None:
        """Обрабатывает изображение и визуализирует результаты"""
        try:
            # Загрузка изображения
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Не удалось загрузить изображение: {image_path}")
            
            # Применение фильтра с сохранением цвета
            filtered_image = self.apply_filter(image, preserve_color=True)
            
            # Получение спектра частот
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            f_transform = fftpack.fft2(gray_image.astype(float))
            f_shift = fftpack.fftshift(f_transform)
            spectrum = self.visualize_spectrum(f_shift)
            
            # Визуализация
            plt.figure(figsize=(15, 5))
            
            # Оригинальное изображение
            plt.subplot(131)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.title('Оригинальное изображение')
            plt.axis('off')
            
            # Спектр частот
            plt.subplot(132)
            plt.imshow(spectrum, cmap='gray')
            plt.title('Спектр частот')
            plt.axis('off')
            
            # Отфильтрованное изображение
            plt.subplot(133)
            plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
            plt.title(f'Отфильтрованное изображение\n(частота среза = {self.cutoff_frequency})')
            plt.axis('off')
            
            plt.tight_layout()
            
            if save_results:
                output_base = Path(image_path).stem
                # Сохранение результатов в соответствующие директории
                plt.savefig(self.analysis_dir / f'{output_base}_analysis.png')
                cv2.imwrite(str(self.filtered_dir / f'{output_base}_filtered.jpg'), filtered_image)
                cv2.imwrite(str(self.spectrum_dir / f'{output_base}_spectrum.jpg'), spectrum)
            
            plt.show()
            logger.info(f"Успешно обработано изображение: {image_path}")
            
        except Exception as e:
            logger.error(f"Ошибка при обработке {image_path}: {str(e)}")

def process_all_images(cutoff_frequency: float = 0.1):
    """
    Обрабатывает все изображения в директории test_images
    
    Args:
        cutoff_frequency: Частота среза фильтра
    """
    image_filter = ImageFilter(cutoff_frequency=cutoff_frequency)
    
    # Получаем список всех изображений в директории test_images
    test_images_dir = Path('test_images')
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    
    image_files = []
    for pattern in image_patterns:
        image_files.extend(test_images_dir.glob(pattern))
    
    if not image_files:
        logger.warning("Изображения не найдены в директории test_images")
        return
    
    logger.info(f"Найдено {len(image_files)} изображений для обработки")
    
    # Обработка каждого изображения
    for image_path in image_files:
        logger.info(f"Обработка изображения: {image_path}")
        image_filter.process_image(image_path)

def main():
    # Задаем частоту среза
    cutoff_frequency = 0.1
    
    try:
        process_all_images(cutoff_frequency)
        logger.info("Обработка всех изображений завершена успешно")
    except Exception as e:
        logger.error(f"Произошла ошибка при обработке изображений: {str(e)}")

if __name__ == "__main__":
    main()