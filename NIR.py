# 0. Проект для НИР кластеризации и дельта кодирования множества схожих изображений.
# 0.1. Импорты.
from itertools import islice
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import imagehash
from PIL import Image
import numpy as np
import time
from contextlib import contextmanager

from matplotlib import pyplot as plt


# 0.2. Функция замера времени выполнения кода
@contextmanager
def timer(block_name="Код"):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        print(f"[{block_name}] Время выполнения: {time.perf_counter() - start_time:.6f} сек.")


# 1. Конфигурация
# 1.1. Размер подвыборки
SUBSET_SIZE = 1000

# 1.2. Веса значимости признаков (wHash, dHash, cHash)
FEATURES_WEIGHT = [0.35, 0.45, 0.2]


# 2. Функции вычислений
# 2.1. Поиск путей подвыборки изображений в датасете
def find_paths(folder_path):
    folder = Path(folder_path)
    files_gen = (str(file) for file in folder.iterdir() if file.is_file())
    return list(islice(files_gen, SUBSET_SIZE))


# 2.2. Вычисление признаков одного изображения (wHash, dHash, cHash)
def compute_features(path):
    image = Image.open(path).convert('RGB')
    return [imagehash.whash(image, hash_size=16),
            imagehash.dhash(image, hash_size=16),
            imagehash.colorhash(image)]


# 2.3. Вычисление взвешанной матрицы расстояний хэмминга
def compute_weighted_hamming_distance_matrix(features):
    features_count = len(features[0])

    distance_tensor = np.zeros((features_count, SUBSET_SIZE, SUBSET_SIZE))

    row_indices, col_indices = np.triu_indices(SUBSET_SIZE, k=1)
    for n in range(features_count):
        for i, j in zip(row_indices, col_indices):
            distance_tensor[n, i, j] = features[i][n] - features[j][n]
            distance_tensor[n, j, i] = distance_tensor[n, i, j]

        distance_tensor[n, :, :] /= distance_tensor[n, :, :].max()

    distance_matrix = np.zeros((SUBSET_SIZE, SUBSET_SIZE))

    assert sum(FEATURES_WEIGHT) <= 1, "Сумма весов признаков больше 1!"

    for n in range(features_count):
        distance_matrix += FEATURES_WEIGHT[n] * distance_tensor[n, :, :]

    return distance_matrix


def image_visual_similarity_plot(arr, image_idxs, similar, top_count):
    n = len(image_idxs)
    fig, axes = plt.subplots(n, (top_count + 1), figsize=(3 * (top_count + 1), 4 * n), squeeze=False)

    for i, idx in enumerate(image_idxs):
        temp_arr = similar[i] * arr[idx]
        top_indices = np.argpartition(temp_arr, top_count)[:top_count]
        top_indices_sorted = top_indices[np.argsort(temp_arr[top_indices])]
        most_diff_paths = [paths[k] for k in top_indices_sorted]

        with Image.open(paths[idx]) as img:
            axes[i, 0].imshow(img)
        axes[i, 0].axis('off')

        for j, path in enumerate(most_diff_paths):
            with Image.open(path) as img:
                axes[i, j + 1].imshow(img)
            axes[i, j + 1].axis('off')

    plt.tight_layout()
    plt.show()


# 3. Запуск программы
# 3.1. Точка входа
if __name__ == "__main__":
    with timer("Поиск путей к изображениям"):
        paths = find_paths("DataSet")

    with timer("Расчёт признаков"):
        with ProcessPoolExecutor(max_workers=None) as ex:
            features = list(ex.map(compute_features, paths, chunksize=10))

    with timer("Расчёт взвешенной матрицы расстояний Хэмминга"):
        WHDM = compute_weighted_hamming_distance_matrix(features)

    image_visual_similarity_plot(WHDM, list(range(0, SUBSET_SIZE, 200)) * 2, [1] * 5 + [-1] * 5, 10)