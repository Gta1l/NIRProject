# 0. Проект для НИР кластеризации и дельта-кодирования множества схожих изображений
# 0.1. Импорты
from DatasetGeneration import *
import lzma
import os
import shutil
from itertools import islice
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import imagehash
import jpeglib
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from sklearn.cluster import HDBSCAN
import zstandard as zstd


# 1. Конфигурация
# 1.1. Общая конфигурация (Размер подвыборки, папки источника/датасета/пулов/отчёта)
SUBSET_SIZE = 0
SRC_DIR = "Source"
OUT_DIR = "Dataset"
JPEG_DIR = "OutJPEG"
POOL_DIR = "OutDeltaPool"
RESTORED_DIR = "OutRestored"
REPORT_DIR = "OutReports"

# 1.2. Размеры итогового изображения, максимальное количество источников
TARGET_WEIGHT = 128
TARGET_HEIGHT = 128
MAX_SOURCES = 100
SEED = 12345

# 1.3. Способы варьирования изображения и их сила
VARIANTS = [
    ("resave", [10, 25, 50, 70]),
    ("noise", [2, 5, 10]),
    ("bright", [4, 10, 20]),
    ("patch", [2, 4, 8]),
    ("shift", [4, 8]),
    ("blur", [1, 2, 3]),
    ("contrast", [10, 25, 40]),
]

# 1.4. Конфигурация вычисления метрик (Веса значимости признаков pHash_Y, pHash_Cr, pHash_Cb, dHash)
FEATURES_WEIGHT = [0.5, 0.2, 0.15, 0.15]
HASH_SIZE = 16

# 1.5. Конфигурация кластеризации
MIN_CLUSTER_SIZE = 5

# 1.6. Конфигурация сохранения файлов
JPEG_QUALITY = 90
DEADZONE_THRESHOLD = 3
ZSTD_LEVEL = 19
ESCAPE_LIMIT = 126

# 1.7. Конфигурация эксперимента
SWEEP = False
DEADZONE_SWEEP = [0, 1, 2, 3, 4, 5, 6, 8, 12, 16]
JPEG_QUALITY_SWEEP = [70, 80, 85, 88, 90, 92, 95]
MIN_CLUSTER_SIZE_SWEEP = [2, 3, 4, 5, 7, 10]
FEATURES_WEIGHT_SWEEP = [
    [0.4, 0.2, 0.2, 0.2],
    [0.25, 0.25, 0.25, 0.25],
    [0.5, 0.2, 0.15, 0.15],
    [0.6, 0.15, 0.15, 0.1],
]

# 1.8. Быстрые параметры упаковки для свипов (не влияют на финальный прогон)
FAST_ZSTD_LEVEL = 3
FAST_LZMA_PRESET = 6

COMPONENT_NAMES = ("Y", "Cb", "Cr")

LZMA_FILTERS = [{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME}]


# 2. Функции
# 2.1. Поиск путей подвыборки SUBSET_SIZE изображений в датасете
def find_paths(folder_path):
    folder = Path(folder_path)
    files = sorted(str(f) for f in folder.iterdir() if f.is_file())
    return files if SUBSET_SIZE <= 0 else list(islice(files, SUBSET_SIZE))


# 2.2. Вычисление признаков одного изображения (бинарный вектор из pHash_Y, pHash_Cr, pHash_Cb, dHash)
def compute_features(path):
    image = Image.open(path).convert("RGB")

    y, cb, cr = image.convert("YCbCr").split()

    half_size = (max(1, image.width // 2), max(1, image.height // 2))
    cb = cb.resize(half_size, Image.Resampling.BOX)
    cr = cr.resize(half_size, Image.Resampling.BOX)

    hashes = [imagehash.phash(y, hash_size=HASH_SIZE),
              imagehash.phash(cr, hash_size=HASH_SIZE),
              imagehash.phash(cb, hash_size=HASH_SIZE),
              imagehash.dhash(y, hash_size=HASH_SIZE)]
    return np.stack([h.hash.flatten() for h in hashes]).astype(np.uint8)


# 2.3. Вычисление взвешенной с FEATURES_WEIGHT матрицы расстояний Жаккара
def compute_weighted_jaccard_distance_matrix(features):
    features = np.asarray(features, dtype=np.float32)
    images_count, features_count, _ = features.shape

    assert abs(sum(FEATURES_WEIGHT) - 1) < 1e-9, "Сумма весов признаков не равна 1!"
    assert len(FEATURES_WEIGHT) == features_count, "Число весов не совпадает с числом признаков!"

    distance_matrix = np.zeros((images_count, images_count), dtype=np.float64)

    for n in range(features_count):
        vectors = features[:, n, :]

        intersection = vectors @ vectors.T
        ones_count = vectors.sum(axis=1)

        union = ones_count[:, None] + ones_count[None, :] - intersection

        similarity = np.where(union > 0, intersection / np.maximum(union, 1), 1.0)
        distance_matrix += FEATURES_WEIGHT[n] * (1.0 - similarity)

    np.fill_diagonal(distance_matrix, 0.0)
    return distance_matrix


# 2.4. Кластеризация с помощью HDBSCAN по матрице расстояний
def clustering(distance_matrix):
    clusterer = HDBSCAN(metric="precomputed", min_cluster_size=MIN_CLUSTER_SIZE, copy=True)
    labels = clusterer.fit_predict(distance_matrix)
    return labels


# 2.5. Нахождение медоидов кластеров (минимум суммы расстояний до членов своего кластера)
def find_medoids(distance_matrix, labels):
    medoids = {}
    for label in sorted(set(labels)):
        if label == -1:
            continue
        members = np.where(labels == label)[0]
        sub = distance_matrix[np.ix_(members, members)]
        best = int(members[int(np.argmin(sub.sum(axis=1)))])
        medoids[int(label)] = best
    return medoids


# 2.6. Сохранение оригиналов в JPEG по кластерам; медоид получает суффикс _base
def save_jpeg(paths, labels, medoids, output_dir, quality=JPEG_QUALITY):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    jpeg_paths = [None] * len(paths)
    jpeg_medoids = {}

    for idx, path in enumerate(paths):
        label = int(labels[idx])
        folder_name = "Noise" if label == -1 else f"Cluster {label}"
        cluster_dir = os.path.join(output_dir, folder_name)
        os.makedirs(cluster_dir, exist_ok=True)

        stem = os.path.splitext(os.path.basename(path))[0]

        if label != -1 and medoids.get(label) == idx:
            save_path = os.path.join(cluster_dir, f"{stem}_base.jpg")
            jpeg_medoids[label] = save_path
        else:
            save_path = os.path.join(cluster_dir, f"{stem}.jpg")

        image = Image.open(path).convert("RGB")
        w, h = image.size
        image = image.crop((0, 0, w // 16 * 16, h // 16 * 16))
        image.save(save_path, "JPEG", quality=quality)
        jpeg_paths[idx] = save_path

    return jpeg_paths, jpeg_medoids


# 2.7. Получение суммарного размера файлов по путям в байтах
def get_total_files_bytes(paths):
    return sum(os.path.getsize(path) for path in paths)


# 2.8. Упаковка байтов: лучший из вариантов «без сжатия / Zstd / LZMA» (первый байт - код упаковщика)
def pack_bytes(data):
    candidates = [b"\x00" + data,
                  b"\x01" + zstd.ZstdCompressor(level=ZSTD_LEVEL).compress(data),
                  b"\x02" + lzma.compress(data, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)]
    return min(candidates, key=len)


def unpack_bytes(blob):
    codec, body = blob[0], blob[1:]
    if codec == 0:
        return body
    if codec == 1:
        return zstd.ZstdDecompressor().decompress(body)
    return lzma.decompress(body, format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)


# 2.9. Подготовка JPEG к хранению: упакованный (.jpgz), только если упаковка действительно выгоднее
def prepare_jpeg_blob(path):
    with open(path, "rb") as f:
        raw = f.read()
    packed = pack_bytes(raw)
    if len(packed) < len(raw):
        return packed, ".jpgz", len(packed)
    return raw, ".jpg", len(raw)


def restore_jpeg_file(stored_path, destination):
    if str(stored_path).endswith(".jpgz"):
        with open(stored_path, "rb") as f:
            data = unpack_bytes(f.read())
        with open(destination, "wb") as f:
            f.write(data)
    else:
        shutil.copy2(stored_path, destination)


# 2.10. Разбор имени файла дельта-пула на исходное имя и роль (base / solo / delta)
def split_stored_name(name):
    for ext in (".jpgz", ".jpg", ".bin"):
        if name.endswith(ext):
            name = name[:-len(ext)]
            break
    for role in ("_base", "_solo", "_delta"):
        if name.endswith(role):
            return name[:-len(role)], role[1:]
    return name, ""


# 2.11. Низкоуровневое чтение JPEG: «сырые» квантованные ДКП-коэффициенты (без декодирования в пиксели)
def read_dct_coefficients(path):
    jpeg = jpeglib.read_dct(path)

    coefficients = []
    for name in COMPONENT_NAMES:
        component = getattr(jpeg, name, None)
        if component is not None:
            coefficients.append(np.array(component, dtype=np.int16))

    quant_tables = np.array(jpeg.qt, dtype=np.int16)

    return jpeg, coefficients, quant_tables


# 2.12. Проверка совместимости Цели и Базы (одинаковые размеры матриц и таблицы квантования)
def is_compatible(base_coefficients, base_quant_tables, target_coefficients, target_quant_tables):
    if len(base_coefficients) != len(target_coefficients):
        return False
    if any(b.shape != t.shape for b, t in zip(base_coefficients, target_coefficients)):
        return False
    return np.array_equal(base_quant_tables, target_quant_tables)


# 2.13. Индексы зигзаг-обхода блока 8x8 (классическая схема JPEG)
def get_zigzag_indices():
    order = sorted(((i, j) for i in range(8) for j in range(8)),
                   key=lambda p: (p[0] + p[1], p[0] if (p[0] + p[1]) % 2 else -p[0]))
    return np.array([i * 8 + j for i, j in order])


ZIGZAG = get_zigzag_indices()


# 2.14. Блоки (Bh, Bw, 8, 8) <-> зигзаг-порядок (число блоков, 64)
def blocks_to_zigzag(blocks):
    return blocks.reshape(-1, 64)[:, ZIGZAG]


def zigzag_to_blocks(zigzag_blocks, shape):
    flat = np.empty_like(zigzag_blocks)
    flat[:, ZIGZAG] = zigzag_blocks
    return flat.reshape(shape)


# 2.15. Оценка цены кодирования блоков (прокси длины кода: логарифм величины + число ненулевых)
def component_block_cost(values):
    a = np.abs(values.astype(np.int32))
    return np.log2(1.0 + a).sum(axis=(-1, -2)) + (a != 0).sum(axis=(-1, -2))


def block_costs(components):
    return np.concatenate([component_block_cost(c).reshape(-1) for c in components])


# 2.16. Кодирование дельты: блочный выбор «дельта / как есть», мёртвая зона, зигзаг, частотная раскладка, упаковка
def encode_delta(target_coefficients, base_coefficients, quant_tables, deadzone):
    modes_parts, values_parts = [], []

    for ci, (target, base) in enumerate(zip(target_coefficients, base_coefficients)):
        target = target.astype(np.int32)
        delta = target - base.astype(np.int32)

        if deadzone > 0:
            qt = quant_tables[min(ci, len(quant_tables) - 1)].astype(np.int32)
            delta = np.where(np.abs(delta) * qt <= deadzone, 0, delta)

        mode_raw = component_block_cost(target) < component_block_cost(delta)
        chosen = np.where(mode_raw[..., None, None], target, delta)

        zigzag_blocks = blocks_to_zigzag(chosen)

        values_parts.append(zigzag_blocks.T.flatten())
        modes_parts.append(mode_raw.flatten())

    modes = np.concatenate(modes_parts)
    values = np.concatenate(values_parts)

    escape_mask = np.abs(values) > ESCAPE_LIMIT
    values8 = np.where(escape_mask, -128, values).astype(np.int8)
    escapes = values[escape_mask].astype(np.int16)

    payload = np.packbits(modes).tobytes() + values8.tobytes() + escapes.tobytes()

    return pack_bytes(payload)


# 2.17. Декодирование дельты: возвращает итоговые коэффициенты Цели
def decode_delta(blob, base_coefficients):
    payload = unpack_bytes(blob)

    total_blocks = sum(c.shape[0] * c.shape[1] for c in base_coefficients)
    total_coefs = total_blocks * 64
    mode_bytes = (total_blocks + 7) // 8

    modes = np.unpackbits(np.frombuffer(payload[:mode_bytes], dtype=np.uint8))[:total_blocks].astype(bool)
    values = np.frombuffer(payload[mode_bytes:mode_bytes + total_coefs], dtype=np.int8).astype(np.int32)
    escapes = np.frombuffer(payload[mode_bytes + total_coefs:], dtype=np.int16)

    escape_pos = values == -128
    assert escape_pos.sum() == len(escapes), "Повреждён поток escape-значений!"
    values[escape_pos] = escapes

    targets = []
    block_offset, value_offset = 0, 0
    for base in base_coefficients:
        block_count = base.shape[0] * base.shape[1]

        zigzag_blocks = np.ascontiguousarray(
            values[value_offset:value_offset + block_count * 64].reshape(64, block_count).T)
        decoded = zigzag_to_blocks(zigzag_blocks, base.shape)

        mode_raw = modes[block_offset:block_offset + block_count].reshape(base.shape[:2])
        target = np.where(mode_raw[..., None, None], decoded, base.astype(np.int32) + decoded)
        targets.append(target.astype(np.int16))

        block_offset += block_count
        value_offset += block_count * 64

    assert block_offset == total_blocks and value_offset == total_coefs
    return targets


# 2.18. Сохранение дельта-пула. Каждый файл хранится в более выгодном виде:
# дельта (.bin) либо JPEG (.jpg / упакованный .jpgz). Медоид кластера сохраняется как База.
def save_cluster_deltas(output_dir, jpeg_paths, labels, medoids, distance_matrix, deadzone):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    medoid_cache = {}
    records = []

    for idx, path in enumerate(jpeg_paths):
        label = int(labels[idx])
        folder_name = "Noise" if label == -1 else f"Cluster {label}"
        cluster_dir = os.path.join(output_dir, folder_name)
        os.makedirs(cluster_dir, exist_ok=True)

        stem = os.path.splitext(os.path.basename(path))[0]
        jpeg_bytes = os.path.getsize(path)
        jpeg_blob, jpeg_ext, jpeg_pack_bytes = prepare_jpeg_blob(path)

        medoid_idx = medoids.get(label)

        if label == -1 or idx == medoid_idx:
            role = "solo" if label == -1 else "base"
            with open(os.path.join(cluster_dir, f"{stem}_{role}{jpeg_ext}"), "wb") as f:
                f.write(jpeg_blob)
            records.append({"stem": stem, "kind": role, "distance": 0.0,
                            "stored_bytes": len(jpeg_blob), "jpeg_bytes": jpeg_bytes,
                            "jpeg_pack_bytes": jpeg_pack_bytes,
                            "jpeg_path": path})
            continue

        if label not in medoid_cache:
            _, base_coefficients, base_quant_tables = read_dct_coefficients(jpeg_paths[medoid_idx])
            medoid_cache[label] = (base_coefficients, base_quant_tables)

        base_coefficients, base_quant_tables = medoid_cache[label]
        _, target_coefficients, target_quant_tables = read_dct_coefficients(path)

        distance = float(distance_matrix[idx, medoid_idx])

        delta_blob = None
        if is_compatible(base_coefficients, base_quant_tables, target_coefficients, target_quant_tables):
            delta_blob = encode_delta(target_coefficients, base_coefficients, target_quant_tables, deadzone)

        if delta_blob is not None and len(delta_blob) < len(jpeg_blob):
            with open(os.path.join(cluster_dir, f"{stem}_delta.bin"), "wb") as f:
                f.write(delta_blob)
            records.append({"stem": stem, "kind": "delta", "distance": distance,
                            "stored_bytes": len(delta_blob), "jpeg_bytes": jpeg_bytes,
                            "jpeg_pack_bytes": jpeg_pack_bytes,
                            "jpeg_path": path})
        else:
            with open(os.path.join(cluster_dir, f"{stem}_solo{jpeg_ext}"), "wb") as f:
                f.write(jpeg_blob)
            records.append({"stem": stem, "kind": "fallback", "distance": distance,
                            "stored_bytes": len(jpeg_blob), "jpeg_bytes": jpeg_bytes,
                            "jpeg_pack_bytes": jpeg_pack_bytes,
                            "jpeg_path": path})

    return records


# 2.19. Восстановление изображений из дельта-пула (возвращает таблицу stem -> путь восстановленного JPEG)
def restore_cluster_deltas(delta_root, output_dir):
    delta_root = Path(delta_root)
    output_root = Path(output_dir)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    restored = {}

    for cluster_dir in delta_root.iterdir():
        if not cluster_dir.is_dir():
            continue

        out_cluster_dir = output_root / cluster_dir.name
        out_cluster_dir.mkdir(parents=True, exist_ok=True)

        base_stem = None
        delta_files = []

        for stored_path in sorted(cluster_dir.iterdir()):
            stem, role = split_stored_name(stored_path.name)
            if role in ("base", "solo"):
                dst = out_cluster_dir / f"{stem}.jpg"
                restore_jpeg_file(stored_path, dst)
                restored[stem] = str(dst)
                if role == "base":
                    base_stem = stem
            elif role == "delta":
                delta_files.append((stem, stored_path))

        if not delta_files:
            continue

        if base_stem is None:
            print(f"[WARN] В {cluster_dir} не найдена База, пропуск.")
            continue

        base_path = str(out_cluster_dir / f"{base_stem}.jpg")
        _, base_coefficients, _ = read_dct_coefficients(base_path)

        with Image.open(base_path) as base_img:
            base_size = base_img.size

        for stem, stored_path in delta_files:
            with open(stored_path, "rb") as f:
                target_coefficients = decode_delta(f.read(), base_coefficients)

            jpeg = jpeglib.read_dct(base_path)
            for name, coefficients in zip(COMPONENT_NAMES, target_coefficients):
                setattr(jpeg, name, coefficients)

            dst = out_cluster_dir / f"{stem}.jpg"
            jpeg.write_dct(str(dst))

            with Image.open(dst) as im:
                if im.size != base_size:
                    im.convert("RGB").resize(base_size, Image.Resampling.LANCZOS).save(
                        dst, "JPEG", quality=JPEG_QUALITY)
            restored[stem] = str(dst)

    return restored


# 2.20. Метрики качества одной пары изображений (PSNR, SSIM)
def compute_quality(reference_arr, test_arr):
    with np.errstate(divide="ignore"):
        psnr = peak_signal_noise_ratio(reference_arr, test_arr, data_range=255)
    ssim = structural_similarity(reference_arr, test_arr, data_range=255, channel_axis=-1)
    return psnr, ssim


def mean_psnr(values):
    finite = [v for v in values if np.isfinite(v)]
    if finite:
        return float(np.mean(finite))
    return float("inf") if values else float("nan")


# 2.21. Оценка сжатия: суммарные коэффициенты и средние SSIM/PSNR (отдельно по дельтам и по всему пулу)
def evaluate_compression(records, restored, quality):
    jpeg_bytes, stored_bytes = 0, 0
    psnr_list, ssim_list = [], []
    psnr_delta, ssim_delta = [], []
    delta_count, skipped, resized = 0, 0, 0

    for record in records:
        jpeg_bytes += record["jpeg_bytes"]
        stored_bytes += record["stored_bytes"]
        if record["kind"] == "delta":
            delta_count += 1

        test_path = restored.get(record["stem"])
        if test_path is None:
            skipped += 1
            continue

        ref_img = Image.open(record["jpeg_path"]).convert("RGB")
        test_img = Image.open(test_path).convert("RGB")

        if test_img.size != ref_img.size:
            test_img = test_img.resize(ref_img.size, Image.Resampling.LANCZOS)
            resized += 1

        reference = np.array(ref_img)
        test = np.array(test_img)

        if reference.shape != test.shape:
            h = min(reference.shape[0], test.shape[0])
            w = min(reference.shape[1], test.shape[1])
            reference = reference[:h, :w]
            test = test[:h, :w]

        try:
            psnr, ssim = compute_quality(reference, test)
        except Exception as e:
            print(f"[SKIP] {record['stem']}: {e}")
            skipped += 1
            continue

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        if record["kind"] == "delta":
            psnr_delta.append(psnr)
            ssim_delta.append(ssim)

    return {
        "quality": quality,
        "files": len(records),
        "delta_files": delta_count,
        "skipped": skipped,
        "resized": resized,
        "jpeg_bytes": jpeg_bytes,
        "stored_bytes": stored_bytes,
        "ratio": jpeg_bytes / stored_bytes if stored_bytes else float("nan"),
        "psnr": mean_psnr(psnr_list),
        "ssim": float(np.mean(ssim_list)) if ssim_list else float("nan"),
        "psnr_delta": mean_psnr(psnr_delta),
        "ssim_delta": float(np.mean(ssim_delta)) if ssim_delta else float("nan"),
    }


# 2.22. Компактная сводка для анализа: кластеры, расстояния, экономия, сравнение с «JPEG+упаковка»
def print_summary(records, labels, medoids, distance_matrix, jpeg_paths):
    label_counts = Counter(int(l) for l in labels)
    clusters = {k: v for k, v in label_counts.items() if k != -1}
    noise = label_counts.get(-1, 0)

    print()
    print("=" * 64)
    print("Сводка по эксперименту")
    print("=" * 64)
    print(f"Кластеров: {len(clusters)} | шум: {noise} | "
          f"размеры кластеров: {sorted(clusters.values(), reverse=True)}")

    intra = []
    for label in clusters:
        members = np.where(np.asarray(labels) == label)[0]
        if len(members) < 2:
            continue
        sub = distance_matrix[np.ix_(members, members)]
        iu = np.triu_indices(len(members), k=1)
        intra.extend(sub[iu].tolist())
    if intra:
        intra = np.asarray(intra)
        print(f"Расстояние внутри кластеров: "
              f"mean={intra.mean():.4f} median={np.median(intra):.4f} "
              f"p90={np.percentile(intra, 90):.4f} max={intra.max():.4f}")

    noise_idx = np.where(np.asarray(labels) == -1)[0]
    if len(noise_idx) and clusters:
        cent_idx = [medoids[l] for l in clusters]
        sub = distance_matrix[np.ix_(noise_idx, cent_idx)]
        nearest = sub.min(axis=1)
        print(f"Шум → ближайший медоид: "
              f"mean={nearest.mean():.4f} median={np.median(nearest):.4f} "
              f"min={nearest.min():.4f} max={nearest.max():.4f}")

    kinds = Counter(r["kind"] for r in records)
    print(f"Роли: base={kinds.get('base', 0)} delta={kinds.get('delta', 0)} "
          f"solo={kinds.get('solo', 0)} fallback={kinds.get('fallback', 0)}")

    jpeg_bytes = sum(r["jpeg_bytes"] for r in records)
    stored_bytes = sum(r["stored_bytes"] for r in records)
    packed_bytes = sum(r["jpeg_pack_bytes"] for r in records)

    print(f"JPEG:            {jpeg_bytes:>10} Б  (1.000)")
    print(f"JPEG + упаковка: {packed_bytes:>10} Б  ({packed_bytes / jpeg_bytes:.3f} от JPEG)")
    print(f"Пул метода:      {stored_bytes:>10} Б  ({stored_bytes / jpeg_bytes:.3f} от JPEG)")
    print(f"Метод vs JPEG:   {jpeg_bytes / stored_bytes:.3f}×")
    print(f"Метод vs JPEG+pack: {packed_bytes / stored_bytes:.3f}×")

    delta_records = [r for r in records if r["kind"] == "delta"]
    if delta_records:
        savings = np.array([(r["jpeg_bytes"] - r["stored_bytes"]) / r["jpeg_bytes"]
                            for r in delta_records])
        print(f"Экономия дельты: mean={savings.mean() * 100:.1f}% "
              f"median={np.median(savings) * 100:.1f}% "
              f"p10={np.percentile(savings, 10) * 100:.1f}% "
              f"p90={np.percentile(savings, 90) * 100:.1f}%")

        dists = np.array([r["distance"] for r in delta_records])
        print(f"Расстояние дельт до медоида: "
              f"mean={dists.mean():.4f} median={np.median(dists):.4f} "
              f"max={dists.max():.4f}")

    fb = [r for r in records if r["kind"] == "fallback"]
    if fb:
        fb_gain = np.array([(r["jpeg_bytes"] - r["stored_bytes"]) / r["jpeg_bytes"] for r in fb])
        print(f"Fallback: {len(fb)} файлов, экономия от упаковки: "
              f"mean={fb_gain.mean() * 100:.1f}% median={np.median(fb_gain) * 100:.1f}%")

    if clusters:
        print(f"Средний размер кластера: "
              f"{np.mean(list(clusters.values())):.1f} файлов "
              f"(всего в кластерах {sum(clusters.values())} из {len(records)})")
    print("=" * 64)
    print()


# 2.23. Графики одного прогона: экономия, расстояния, распределение пула по ролям
def plot_single_run(records, labels, distance_matrix, medoids, report, output_dir=REPORT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    delta = [r for r in records if r["kind"] == "delta"]
    if not delta:
        return

    dists = np.array([r["distance"] for r in delta])
    savings = np.array([(r["jpeg_bytes"] - r["stored_bytes"]) / r["jpeg_bytes"] * 100
                        for r in delta])
    ratios = np.array([r["stored_bytes"] / r["jpeg_bytes"] for r in delta])

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].scatter(dists, savings, s=10, alpha=0.6)
    axes[0].set_xlabel("Расстояние до медоида")
    axes[0].set_ylabel("Экономия, %")
    axes[0].set_title("Дельта: экономия vs расстояние")
    axes[0].grid(alpha=0.3)

    axes[1].hist(savings, bins=30, alpha=0.75)
    axes[1].axvline(np.median(savings), color="red", linestyle="--",
                    label=f"медиана {np.median(savings):.1f}%")
    axes[1].set_xlabel("Экономия, %")
    axes[1].set_ylabel("Файлов")
    axes[1].set_title("Распределение экономии дельты")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].scatter(dists, ratios, s=10, alpha=0.6, color="tab:orange")
    axes[2].set_xlabel("Расстояние до медоида")
    axes[2].set_ylabel("Размер дельты / JPEG")
    axes[2].set_title("Дельта: относительный размер vs расстояние")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "single_run_scatter.png"), dpi=200)
    plt.close(fig)

    kinds_bytes = {}
    for r in records:
        kinds_bytes[r["kind"]] = kinds_bytes.get(r["kind"], 0) + r["stored_bytes"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    names = list(kinds_bytes.keys())
    sizes_kb = [kinds_bytes[k] / 1024 for k in names]
    palette = {"base": "tab:blue", "delta": "tab:orange",
               "solo": "tab:green", "fallback": "tab:red"}
    axes[0].bar(names, sizes_kb, color=[palette.get(n, "gray") for n in names])
    axes[0].set_ylabel("Размер, КБ")
    axes[0].set_title("Пул по ролям")
    axes[0].grid(alpha=0.3, axis="y")

    noise_idx = np.where(np.asarray(labels) == -1)[0]
    if len(noise_idx) and medoids:
        cent_idx = [medoids[l] for l in medoids]
        sub = distance_matrix[np.ix_(noise_idx, cent_idx)]
        nearest = sub.min(axis=1)
        axes[1].hist(nearest, bins=20, alpha=0.75, color="tab:red")
        axes[1].set_xlabel("Расстояние до ближайшего медоида")
        axes[1].set_ylabel("Файлов")
        axes[1].set_title(f"Шум ({len(noise_idx)} файлов)")
        axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "single_run_roles.png"), dpi=200)
    plt.close(fig)


# 2.24. Графики серии прогонов: метрики и сжатие vs параметр свипа
def plot_sweep(runs, param_name="параметр", output_dir=REPORT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    if len(runs) < 2:
        print("[plot_sweep] нужно минимум два прогона.")
        return

    xs = [str(r["param"]) for r in runs]
    ratios = [r["report"]["ratio"] for r in runs]
    psnr = [r["report"].get("psnr_delta", r["report"]["psnr"]) for r in runs]
    ssim = [r["report"].get("ssim_delta", r["report"]["ssim"]) for r in runs]
    deltas = [r["report"].get("delta_files", 0) for r in runs]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    axes[0].plot(xs, ratios, "o-")
    axes[0].set_xlabel(param_name)
    axes[0].set_ylabel("Метод / JPEG")
    axes[0].set_title("Коэффициент сжатия")
    axes[0].grid(alpha=0.3)
    axes[0].tick_params(axis="x", rotation=30)

    axes[1].plot(xs, psnr, "s-", color="tab:green")
    axes[1].set_xlabel(param_name)
    axes[1].set_ylabel("PSNR дельт, дБ")
    axes[1].set_title("PSNR (только дельты)")
    axes[1].grid(alpha=0.3)
    axes[1].tick_params(axis="x", rotation=30)

    axes[2].plot(xs, ssim, "^-", color="tab:orange")
    axes[2].set_xlabel(param_name)
    axes[2].set_ylabel("SSIM дельт")
    axes[2].set_title("SSIM (только дельты)")
    axes[2].grid(alpha=0.3)
    axes[2].tick_params(axis="x", rotation=30)

    axes[3].plot(ratios, psnr, "o-", color="tab:red")
    for x, y, r in zip(ratios, psnr, runs):
        axes[3].annotate(str(r["param"]), (x, y), fontsize=8,
                         xytext=(3, 3), textcoords="offset points")
    axes[3].set_xlabel("Метод / JPEG")
    axes[3].set_ylabel("PSNR дельт, дБ")
    axes[3].set_title("Rate–distortion")
    axes[3].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"sweep_{param_name}.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(xs, deltas, "d-", color="tab:purple")
    ax.set_xlabel(param_name)
    ax.set_ylabel("Файлов дельтой")
    ax.set_title("Покрытие методом")
    ax.grid(alpha=0.3)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"sweep_{param_name}_coverage.png"), dpi=200)
    plt.close(fig)


# 2.25. Сводный rate-distortion по нескольким свипам
def plot_rd_overview(all_sweeps, output_dir=REPORT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 6))
    markers = ["o", "s", "^", "D", "v", "P", "*"]

    for i, (name, runs) in enumerate(all_sweeps.items()):
        xs = [r["report"]["ratio"] for r in runs]
        ys = [r["report"].get("psnr_delta", r["report"]["psnr"]) for r in runs]
        ax.plot(xs, ys, marker=markers[i % len(markers)], label=name, alpha=0.85)

    ax.set_xlabel("Метод / JPEG (ниже — лучше сжатие)")
    ax.set_ylabel("PSNR дельт, дБ")
    ax.set_title("Rate–distortion: сводка по всем свипам")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "rd_all_sweeps.png"), dpi=200)
    plt.close(fig)


# 2.26. Прогон полного конвейера с опциональным переопределением глобального конфига
def run_pipeline(overrides=None, regenerate=True, fast=False):
    saved = {}
    effective = dict(overrides or {})
    if fast:
        effective.setdefault("ZSTD_LEVEL", FAST_ZSTD_LEVEL)
        effective.setdefault("LZMA_FILTERS",
                             [{"id": lzma.FILTER_LZMA2, "preset": FAST_LZMA_PRESET}])

    for key, value in effective.items():
        saved[key] = globals()[key]
        globals()[key] = value

    try:
        if regenerate:
            generate(SRC_DIR, OUT_DIR, TARGET_WEIGHT, TARGET_HEIGHT,
                     MAX_SOURCES, VARIANTS, SEED)

        paths = find_paths(OUT_DIR)

        with ProcessPoolExecutor(max_workers=None) as ex:
            features = list(ex.map(compute_features, paths, chunksize=10))

        distance_matrix = compute_weighted_jaccard_distance_matrix(features)
        labels = clustering(distance_matrix)
        medoids = find_medoids(distance_matrix, labels)

        jpeg_paths, _ = save_jpeg(paths, labels, medoids, JPEG_DIR, JPEG_QUALITY)
        records = save_cluster_deltas(POOL_DIR, jpeg_paths, labels, medoids,
                                      distance_matrix, DEADZONE_THRESHOLD)
        restored = restore_cluster_deltas(POOL_DIR, RESTORED_DIR)
        report = evaluate_compression(records, restored, JPEG_QUALITY)

        return {
            "records": records,
            "labels": labels,
            "medoids": medoids,
            "distance_matrix": distance_matrix,
            "jpeg_paths": jpeg_paths,
            "report": report,
        }
    finally:
        for key, value in saved.items():
            globals()[key] = value


# 2.27. Серия прогонов с вариацией одного параметра конфига
def run_sweep(param_name, values, overrides=None, regenerate=False, fast=True):
    runs = []
    for value in values:
        print(f"\n--- {param_name} = {value} ---")
        current = {param_name: value}
        if overrides:
            current.update(overrides)
        result = run_pipeline(current, regenerate=regenerate, fast=fast)
        result["param"] = value
        result["param_name"] = param_name
        runs.append(result)
    return runs


# 3. Запуск программы
# 3.1. Точка входа
if __name__ == "__main__":

    # 3.1.1. Одиночный прогон основной конфигурации (полное качество упаковки)
    with timer("Одиночный прогон конвейера"):
        run = run_pipeline(regenerate=True, fast=False)

    with timer("Сводка для анализа"):
        print_summary(run["records"], run["labels"], run["medoids"],
                      run["distance_matrix"], run["jpeg_paths"])

    report = run["report"]
    print(f"Файлов: {report['files']}, дельтами сохранено: {report['delta_files']}")
    print(f"JPEG: {report['jpeg_bytes']} Б, пул: {report['stored_bytes']} Б, "
          f"коэффициент: {report['ratio']:.3f}")
    print(f"PSNR всего: {report['psnr']:.2f} дБ, SSIM всего: {report['ssim']:.4f}")
    print(f"PSNR дельт: {report['psnr_delta']:.2f} дБ, SSIM дельт: {report['ssim_delta']:.4f}")

    with timer("Графики одиночного прогона"):
        plot_single_run(run["records"], run["labels"], run["distance_matrix"],
                        run["medoids"], run["report"])

    if (SWEEP):
        # 3.1.2. Свип по DEADZONE_THRESHOLD
        with timer("Свип по DEADZONE_THRESHOLD"):
            sweep_dz = run_sweep("DEADZONE_THRESHOLD", DEADZONE_SWEEP)
            plot_sweep(sweep_dz, "DEADZONE_THRESHOLD")

        # 3.1.3. Свип по JPEG_QUALITY
        with timer("Свип по JPEG_QUALITY"):
            sweep_q = run_sweep("JPEG_QUALITY", JPEG_QUALITY_SWEEP)
            plot_sweep(sweep_q, "JPEG_QUALITY")

        # 3.1.4. Свип по MIN_CLUSTER_SIZE
        with timer("Свип по MIN_CLUSTER_SIZE"):
            sweep_mcs = run_sweep("MIN_CLUSTER_SIZE", MIN_CLUSTER_SIZE_SWEEP)
            plot_sweep(sweep_mcs, "MIN_CLUSTER_SIZE")

        # 3.1.5. Свип по FEATURES_WEIGHT
        with timer("Свип по FEATURES_WEIGHT"):
            sweep_w = []
            for w in FEATURES_WEIGHT_SWEEP:
                print(f"\n--- FEATURES_WEIGHT = {w} ---")
                result = run_pipeline({"FEATURES_WEIGHT": list(w)},
                                      regenerate=False, fast=True)
                result["param"] = "-".join(f"{v:.2f}" for v in w)
                result["param_name"] = "FEATURES_WEIGHT"
                sweep_w.append(result)
            plot_sweep(sweep_w, "FEATURES_WEIGHT")

        # 3.1.6. Сводный rate-distortion по всем свипам
        with timer("Сводный rate-distortion"):
            plot_rd_overview({
                "DEADZONE":      sweep_dz,
                "JPEG_QUALITY":  sweep_q,
                "MIN_CLUSTER":   sweep_mcs,
                "FEATURES_W":    sweep_w,
            })

        # 3.1.7. Итог
        print("\nВсе графики сохранены в", os.path.abspath(REPORT_DIR))
        for name in ("single_run_scatter.png", "single_run_roles.png",
                     "sweep_DEADZONE_THRESHOLD.png", "sweep_DEADZONE_THRESHOLD_coverage.png",
                     "sweep_JPEG_QUALITY.png", "sweep_JPEG_QUALITY_coverage.png",
                     "sweep_MIN_CLUSTER_SIZE.png", "sweep_MIN_CLUSTER_SIZE_coverage.png",
                     "sweep_FEATURES_WEIGHT.png", "sweep_FEATURES_WEIGHT_coverage.png",
                     "rd_all_sweeps.png"):
            print(f"  {name}")