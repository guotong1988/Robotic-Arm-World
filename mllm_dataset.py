"""把各任务的样本并进同一个目录，给训练读。

训练看 samples.jsonl 里的：image、image_back、instruction、answer。
image 是 frontview，image_back 是 backview（frontview 正对面，双目时才有），
都是相对这个目录的路径。instruction 以每台相机一个 <image> 占位开头。
task 用来区分任务，也用来重跑时替换旧样本。
只采一个任务时，meta.json 仍是原来的扁平格式，评测能直接读相机矩阵。
两个任务都在时，各自的统计和相机矩阵放在 meta.json 的 tasks 下面。
"""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
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


def _row_images(row):
    """一条样本引用的所有图片相对路径：frontview、backview（有的话）。"""
    return [rel for rel in (row.get("image"), row.get("image_back")) if rel]


def _prune_images(out_dir, task, merged, dropped_images):
    image_dir = out_dir / "images"
    keep_names = set()
    for row in merged:
        for rel in _row_images(row):
            keep_names.add(Path(rel).name)
    for rel in dropped_images:
        path = out_dir / rel
        if path.parent == image_dir and path.name not in keep_names and path.is_file():
            path.unlink()
    if image_dir.is_dir():
        for path in image_dir.glob(task + "_*.png"):
            if path.name not in keep_names:
                path.unlink()


def make_image_staging_dir(out_dir, task):
    """正式图片发布前的任务私有目录，避免残缺采集覆盖同名旧图。"""
    out_dir = Path(out_dir)
    staging_root = out_dir / ".collect_pending"
    staging_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=task + "_", dir=str(staging_root)))


def finish_image_staging(out_dir, staging_dir, publish):
    staging_dir = Path(staging_dir)
    if publish:
        image_dir = Path(out_dir) / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for source in staging_dir.glob("*.png"):
            os.replace(str(source), str(image_dir / source.name))
    shutil.rmtree(staging_dir)
    try:
        staging_dir.parent.rmdir()
    except OSError:
        pass


def _lock_path(out_dir):
    """锁放在本地文件系统。--out 常常是 OSS/FUSE 挂载，不支持 flock（Errno 95）。

    采集进程都在同一台机器上跑，按输出目录的绝对路径做哈希，同一目录的写仍然互斥。
    带上 uid，避免 /tmp 下不同用户的锁文件互相没权限。
    """
    digest = hashlib.sha1(os.path.abspath(str(out_dir)).encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / "mllm_publish_{}_{}.lock".format(os.getuid(), digest)


def publish_task(out_dir, task, samples, task_meta):
    """写入本任务样本，保留目录里其他任务。返回合并后的样本数。

    多个采集进程并行写同一个 --out，先拿文件锁再读改写，
    避免 samples.jsonl / meta.json 被并发覆盖丢样本。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with _lock_path(out_dir).open("w") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        except OSError as exc:
            print("publish_task: 锁文件不支持 flock（{}），不加锁写入".format(exc), flush=True)
        return _publish_task(out_dir, task, samples, task_meta)


def _publish_task(out_dir, task, samples, task_meta):
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
                dropped_images.extend(_row_images(row))
                continue
            row = dict(row)
            row["task"] = legacy_env
            row_task = legacy_env
        if row_task == task:
            dropped_images.extend(_row_images(row))
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
        "inputs": _shared_value(bodies, "inputs") or ["image", "image_back", "instruction"],
        "output": {
            "answer": "JSON 字符串，带 subtasks，每个 subtask 带一句 desc；字段说明跟在指令话术后面。训练读 samples.jsonl 的 image、instruction、answer；task 是小写的环境名，各任务的步骤说明在 tasks 里。",
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


def _shard_meta(out_dir, task):
    meta = _read_json(out_dir / "meta.json")
    if "tasks" in meta:
        if task not in meta["tasks"]:
            raise ValueError("{} 的 meta.json 缺少任务 {}".format(out_dir, task))
        return meta["tasks"][task]
    if not meta:
        raise ValueError("{} 缺少有效的 meta.json".format(out_dir))
    return meta


def _meta_config(meta):
    volatile = {"num_episodes", "num_discarded_episodes", "num_samples", "episodes"}
    return {key: value for key, value in meta.items() if key not in volatile}


def merge_task_shards(out_dir, task, shard_dirs, expected_episodes):
    """校验并合并同任务的独立采集分片；校验失败时不碰正式数据。"""
    if not shard_dirs:
        raise ValueError("至少需要一个采集分片")
    out_dir = Path(out_dir)
    rows = []
    episodes = []
    sample_ids = set()
    episode_ids = set()
    image_paths = set()
    copies = []
    merged_meta = None
    meta_config = None
    discarded = 0

    for shard_dir in shard_dirs:
        shard_dir = Path(shard_dir)
        shard_rows = _read_jsonl(shard_dir / "samples.jsonl")
        if not shard_rows:
            raise ValueError("分片 {} 没有样本，拒绝发布".format(shard_dir))
        meta = _shard_meta(shard_dir, task)
        config = _meta_config(meta)
        if meta_config is None:
            merged_meta = dict(meta)
            meta_config = config
        elif config != meta_config:
            raise ValueError("分片 {} 的相机或任务元数据与其他分片不一致".format(shard_dir))

        shard_episodes = set()
        for row in shard_rows:
            if row.get("task") != task:
                raise ValueError("分片 {} 含其他任务的样本".format(shard_dir))
            sample_id = row.get("id")
            episode = row.get("episode")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError("分片 {} 含无效样本 ID".format(shard_dir))
            if not isinstance(episode, int) or isinstance(episode, bool):
                raise ValueError("样本 {} 的 episode 无效".format(sample_id))
            if sample_id in sample_ids:
                raise ValueError("样本 ID 重复：{}".format(sample_id))
            if episode in episode_ids:
                raise ValueError("episode 重复：{}".format(episode))
            sample_ids.add(sample_id)
            episode_ids.add(episode)
            shard_episodes.add(episode)
            rows.append(row)
            for rel in _row_images(row):
                rel_path = Path(rel)
                if rel_path.is_absolute() or not rel_path.parts or rel_path.parts[0] != "images" or ".." in rel_path.parts:
                    raise ValueError("样本 {} 的图片路径不安全：{}".format(sample_id, rel))
                source = shard_dir / rel_path
                if not source.is_file():
                    raise ValueError("样本 {} 缺少图片：{}".format(sample_id, source))
                if rel in image_paths:
                    raise ValueError("多个样本引用同一图片：{}".format(rel))
                image_paths.add(rel)
                copies.append((source, out_dir / rel_path))

        if meta.get("num_episodes") != len(shard_episodes):
            raise ValueError("分片 {} 的 episode 统计与样本不一致".format(shard_dir))
        if meta.get("num_samples") != len(shard_rows):
            raise ValueError("分片 {} 的样本统计与 samples.jsonl 不一致".format(shard_dir))
        shard_meta_episodes = meta.get("episodes", [])
        meta_episode_ids = [entry.get("episode") for entry in shard_meta_episodes]
        if len(meta_episode_ids) != len(shard_episodes) or set(meta_episode_ids) != shard_episodes:
            raise ValueError("分片 {} 的 meta episodes 与样本编号不一致".format(shard_dir))
        episodes.extend(shard_meta_episodes)
        discarded += int(meta.get("num_discarded_episodes", 0))

    if len(episode_ids) != int(expected_episodes):
        raise ValueError(
            "任务 {} 只有 {} / {} 条成功轨迹，拒绝发布".format(task, len(episode_ids), expected_episodes)
        )
    if len(episodes) != len(episode_ids):
        raise ValueError("任务 {} 的 meta episodes 与样本数量不一致".format(task))

    for _, destination in copies:
        destination.parent.mkdir(parents=True, exist_ok=True)
    for source, destination in copies:
        shutil.copy2(source, destination)
    rows.sort(key=lambda row: (row["episode"], row.get("step", 0), row["id"]))
    merged_meta["num_episodes"] = len(episode_ids)
    merged_meta["num_discarded_episodes"] = discarded
    merged_meta["num_samples"] = len(rows)
    merged_meta["episodes"] = sorted(episodes, key=lambda entry: entry["episode"])
    return publish_task(out_dir, task, rows, merged_meta)


def main():
    parser = argparse.ArgumentParser(description="MLLM 数据集的分片合并工具")
    subparsers = parser.add_subparsers(dest="command", required=True)
    merge = subparsers.add_parser("merge-task", help="全部校验通过后合并一个任务的采集分片")
    merge.add_argument("--out", required=True, help="正式数据集目录")
    merge.add_argument("--task", required=True)
    merge.add_argument("--episodes", type=int, required=True, help="必须达到的成功轨迹总数")
    merge.add_argument("--shards", nargs="+", required=True, help="该任务的独立分片目录")
    args = parser.parse_args()
    if args.command == "merge-task":
        count = merge_task_shards(args.out, args.task, args.shards, args.episodes)
        print("published task {} with {} episodes; {} samples in {}".format(
            args.task, args.episodes, count, args.out
        ))


if __name__ == "__main__":
    main()
