# 0. Вспомогательный модуль генерации датасета схожих изображений
# 0.1. Импорты.
import io
import shutil
import time
from pathlib import Path
import numpy as np
from PIL import Image
from contextlib import contextmanager


# 0.2. Функция замера времени выполнения кода
@contextmanager
def timer(block_name="Код"):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        print(f"[{block_name}] Время выполнения: {time.perf_counter() - start_time:.6f} сек.")


# 2. Основные функции
# 2.1. Стандартизация изображений под размеры (TARGET_WEIGHT, TARGET_HEIGHT)
def standardize_image(path, target_size=(256, 256)):
    image = Image.open(path).convert("RGB")
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    return np.array(image)


# 2.2. Варьирование изображений
def make_variant(arr, kind, strength, rng):
    if kind == "resave":
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, "JPEG", quality=100 - strength)
        return np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))

    if kind == "noise":
        noisy = arr.astype(np.float32) + rng.normal(0.0, strength, arr.shape)
        return np.clip(np.rint(noisy), 0, 255).astype(np.uint8)

    if kind == "bright":
        return np.clip(arr.astype(np.int16) + strength, 0, 255).astype(np.uint8)

    if kind == "contrast":
        factor = 1.0 + strength / 100.0
        out = (arr.astype(np.float32) - 127.5) * factor + 127.5
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)

    if kind == "patch":
        out = arr.copy()
        h, w = out.shape[:2]
        side = max(4, int(((strength / 100.0) * w * h / 4) ** 0.5))
        for _ in range(4):
            y = int(rng.integers(0, max(1, h - side)))
            x = int(rng.integers(0, max(1, w - side)))
            out[y:y + side, x:x + side] = rng.integers(0, 256, 3)
        return out

    if kind == "shift":
        padded = np.pad(arr, ((0, strength), (0, strength), (0, 0)), mode="edge")
        return padded[strength:, strength:]

    if kind == "blur":
        r = int(strength)
        k = 2 * r + 1
        padded = np.pad(arr.astype(np.float32), ((r, r), (r, r), (0, 0)), mode="edge")
        cum = padded.cumsum(0).cumsum(1)
        cum = np.pad(cum, ((1, 0), (1, 0), (0, 0)), mode="constant")
        s = (cum[k:, k:] - cum[:-k, k:] - cum[k:, :-k] + cum[:-k, :-k])
        out = s / (k * k)
        return np.clip(np.rint(out), 0, 255).astype(np.uint8)

    raise ValueError(f"Неизвестный тип искажения: {kind}")


# 2.3. Функция генерации
def generate(source, out, target_w, target_h, max_sources, variants, seed):
    rng = np.random.default_rng(seed)

    src = Path(source)
    if not src.exists():
        raise SystemExit(f"Папка {source} не найдена.")

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    source_paths = sorted(p for p in src.iterdir()
                          if p.is_file() and p.suffix.lower() in exts)
    if not source_paths:
        raise SystemExit(f"В папке {source} нет изображений.")

    rng.shuffle(source_paths)
    if max_sources > 0:
        source_paths = source_paths[:max_sources]

    out_dir = Path(out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for group_id, path in enumerate(source_paths):
        try:
            arr = standardize_image(path, (target_w, target_h))
        except Exception as error:
            print(f"[WARNING] {path.name}: {error}, пропуск.")
            continue

        stem = f"{group_id:04d}_orig"
        Image.fromarray(arr).save(out_dir / f"{stem}.png")
        count += 1

        for kind, strengths in variants:
            for strength in strengths:
                stem = f"{group_id:04d}_{kind}_{strength}"
                variant = make_variant(arr, kind, strength, rng)
                Image.fromarray(variant).save(out_dir / f"{stem}.png")
                count += 1

    print(f"Создано файлов: {count} в папке {out}")
    print(f"Размер всех изображений: {target_w} x {target_h}")