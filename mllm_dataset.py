"""把 Lift / Stack 的样本并进同一个目录，给训练读。

训练只看 samples.jsonl 里的三列：image、instruction、answer。
image 是相对这个目录的路径。task 用来区分任务，也用来重跑时替换旧样本。
只采一个任务时，meta.json 仍是原来的扁平格式，评测能直接读相机矩阵。
两个任务都在时，各自的统计和相机矩阵放在 meta.json 的 tasks 下面。
"""

import json
from pathlib import Path


def make_sample_id(task, episode, phase, step):
    return "{}_ep{:04d}_{}_{:04d}".format(task, int(episode), phase, int(step))


def _read_json(path):
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path):
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _shared_value(bodies, key):
    values = [body.get(key) for body in bodies]
    if values and all(value == values[0] for value in values):
        return values[0]
    return None


def _prune_images(out_dir, task, merged, dropped_images):
    image_dir = out_dir / "images"
    keep_names = set()
    for row in merged:
        image = row.get("image")
        if image:
            keep_names.add(Path(image).name)
    for rel in dropped_images:
        path = out_dir / rel
        if path.parent == image_dir and path.name not in keep_names and path.is_file():
            path.unlink()
    if image_dir.is_dir():
        for path in image_dir.glob(task + "_*.png"):
            if path.name not in keep_names:
                path.unlink()


def publish_task(out_dir, task, samples, task_meta):
    """写入本任务样本，保留目录里其他任务。返回合并后的样本数。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "samples.jsonl"
    meta_path = out_dir / "meta.json"
    existing_meta = _read_json(meta_path)
    legacy_env = None
    if "tasks" not in existing_meta and existing_meta.get("env"):
        legacy_env = str(existing_meta["env"]).strip().lower()

    owned = []
    for record in samples:
        row = dict(record)
        row["task"] = task
        owned.append(row)

    kept = []
    dropped_images = []
    for row in _read_jsonl(jsonl_path):
        row_task = row.get("task")
        if row_task is None and legacy_env:
            if legacy_env == task:
                if row.get("image"):
                    dropped_images.append(row["image"])
                continue
            row = dict(row)
            row["task"] = legacy_env
            row_task = legacy_env
        if row_task == task:
            if row.get("image"):
                dropped_images.append(row["image"])
            continue
        kept.append(row)

    merged = kept + owned
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in merged:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    _prune_images(out_dir, task, merged, dropped_images)

    task_meta = dict(task_meta)
    task_meta["task"] = task
    tasks = {}
    if isinstance(existing_meta.get("tasks"), dict):
        tasks.update(existing_meta["tasks"])
    elif legacy_env and legacy_env != task:
        previous = dict(existing_meta)
        previous.setdefault("task", legacy_env)
        tasks[legacy_env] = previous
    if owned:
        tasks[task] = task_meta
    else:
        tasks.pop(task, None)
    present = {row.get("task") for row in merged}
    tasks = {name: body for name, body in tasks.items() if name in present}

    if len(tasks) <= 1:
        if len(tasks) == 1:
            flat = dict(next(iter(tasks.values())))
        else:
            flat = dict(task_meta)
        flat["num_samples"] = len(merged)
        meta_path.write_text(json.dumps(flat, ensure_ascii=False, indent=2), encoding="utf-8")
        return len(merged)

    bodies = list(tasks.values())
    combined = {
        "inputs": ["image", "instruction"],
        "output": {
            "answer": "JSON 字符串，只有 subtasks。训练读 samples.jsonl 的 image、instruction、answer；task 是小写的环境名，各任务的步骤说明在 tasks 里。",
        },
        "num_samples": len(merged),
        "num_episodes": sum(int(body.get("num_episodes") or 0) for body in bodies),
        "tasks": tasks,
    }
    camera = _shared_value(bodies, "camera")
    image_size = _shared_value(bodies, "image_size")
    matrix = _shared_value(bodies, "camera_world_to_pixel")
    if camera is not None:
        combined["camera"] = camera
    if image_size is not None:
        combined["image_size"] = image_size
    if matrix is not None:
        combined["camera_world_to_pixel"] = matrix
    meta_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(merged)
