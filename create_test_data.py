import os
import shutil
import random


def split_data(source_images, source_annotations, dest_images, dest_annotations, test_size=0.2):
    # Получение списка файлов
    files = os.listdir(source_images)
    # Перемешивание списка
    random.shuffle(files)

    # Определение количества тестовых файлов
    num_test_files = int(len(files) * test_size)

    # Создание папок, если они еще не существуют
    os.makedirs(dest_images, exist_ok=True)
    os.makedirs(dest_annotations, exist_ok=True)

    # Перемещение файлов
    for i in range(num_test_files):
        image_file = files[i]
        annotation_file = image_file.replace('.jpg', '.txt').replace('.png',
                                                                     '.txt')  # Замените расширения, если они отличаются

        # Перемещение изображений и аннотаций в тестовые папки
        shutil.move(os.path.join(source_images, image_file), os.path.join(dest_images, image_file))
        shutil.move(os.path.join(source_annotations, annotation_file), os.path.join(dest_annotations, annotation_file))


# Пути к исходным и целевым папкам
source_images = '/home/aleksandr/dataset/clips/all_images'
source_annotations = '/home/aleksandr/dataset/clips/all_annotations'
dest_images = '/home/aleksandr/dataset/clips/test_images'
dest_annotations = '/home/aleksandr/dataset/clips/test_annotations'

# Вызов функции
split_data(source_images, source_annotations, dest_images, dest_annotations, test_size=0.2)
