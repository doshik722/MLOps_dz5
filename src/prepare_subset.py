"""Воспроизводимый стратифицированный срез train из готовых тензоров ДЗ 4.

Исходный train.pt и весь val.pt сохраняются. Токены и labels не меняются.
"""

import hashlib
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.config import load_params


def main() -> None:
    params = load_params()
    cfg = params["data"]
    source, out = Path(cfg["source_train"]), Path(cfg["train"])
    if source.resolve() == out.resolve():
        raise SystemExit("Срез должен сохраняться отдельно от исходного train.pt")
    blob = torch.load(source, map_location="cpu", weights_only=False)
    examples = blob["examples"]
    size, seed = cfg["subset_size"], cfg["subset_seed"]
    if not 0 < size <= len(examples):
        raise SystemExit(f"subset_size должен быть от 1 до {len(examples)}")
    if blob["model"] != params["model"]["name"]:
        raise SystemExit("Модель токенизации ДЗ 4 не совпадает с моделью обучения")
    tok = AutoTokenizer.from_pretrained(params["model"]["name"], local_files_only=True)
    groups = defaultdict(list)
    expected = {"негативное", "нейтральное", "позитивное"}
    excluded = Counter()
    for index, example in enumerate(examples):
        label = tok.decode([t for t in example["labels"] if t != -100], skip_special_tokens=True).strip()
        # После обрезки в ДЗ 4 иногда остаётся лишь начало метки: «нег», «ней».
        # Исходные тензоры не исправляем; в срез берём только полные ответы.
        if label not in expected:
            excluded[label] += 1
            continue
        groups[label].append(index)
    if set(groups) != expected:
        raise SystemExit(f"Неожиданные метки тональности: {sorted(groups)}")
    eligible = sum(len(indices) for indices in groups.values())
    if size > eligible:
        raise SystemExit(f"Полных ответов {eligible}, для подвыборки {size} недостаточно")
    # Пропорции классов исходного train сохраняются методом наибольших остатков.
    exact = {label: size * len(indices) / eligible for label, indices in groups.items()}
    quotas = {label: math.floor(count) for label, count in exact.items()}
    remainder = size - sum(quotas.values())
    for label in sorted(groups, key=lambda key: (-(exact[key] - quotas[key]), key))[:remainder]:
        quotas[label] += 1
    rng = random.Random(seed)
    selected = []
    for label in sorted(groups):
        selected.extend(rng.sample(groups[label], quotas[label]))
    rng.shuffle(selected)
    result = dict(blob)
    result["examples"] = [examples[i] for i in selected]
    result["subset"] = {
        "source": source.as_posix(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_examples": len(examples),
        "eligible_examples": eligible,
        "excluded_incomplete_labels": dict(sorted(excluded.items())),
        "selected_examples": size,
        "seed": seed,
        "source_counts": dict(sorted((label, len(indices)) for label, indices in groups.items())),
        "selected_counts": dict(sorted(quotas.items())),
        "source_indices": selected,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, out)
    print(f"Train: {size} из {len(examples)}, seed={seed}, классы={quotas}")
    print(f"Полных ответов: {eligible}; не включены обрезанные метки: {dict(excluded)}")
    print(f"-> {out}; исходный train и val не изменены")


if __name__ == "__main__":
    main()
