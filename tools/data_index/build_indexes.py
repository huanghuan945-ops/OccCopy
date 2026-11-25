"""Index nuScenes trainval metadata and an uncompressed Occ3D label archive."""
import argparse
import io
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[2]


def fingerprint(path):
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def json_items(binary):
    """Stream a JSON array without holding the 1.3GB sample_data table in RAM."""
    f = io.TextIOWrapper(binary, encoding='utf-8')
    decoder = json.JSONDecoder()
    buf, pos, eof, started = '', 0, False, False
    while True:
        if pos >= len(buf) or len(buf) - pos < 4096:
            block = f.read(1 << 20)
            buf, pos, eof = buf[pos:] + block, 0, not block
        while pos < len(buf) and (buf[pos].isspace() or buf[pos] == ','):
            pos += 1
        if not started:
            if pos == len(buf):
                if eof:
                    raise ValueError('empty JSON table')
                continue
            if buf[pos] != '[':
                raise ValueError('not an array')
            started, pos = True, pos + 1
            continue
        if pos < len(buf) and buf[pos] == ']':
            return
        try:
            value, end = decoder.raw_decode(buf, pos)
        except json.JSONDecodeError:
            if eof:
                raise
            block = f.read(1 << 20)
            buf, pos, eof = buf[pos:] + block, 0, not block
            continue
        pos = end
        yield value


def metadata_index(path):
    with tarfile.open(path, "r:*") as archive:
        members = {Path(member.name).stem: member for member in archive
                   if member.isfile() and member.name.endswith(".json")}
        tables = {}
        for name in ("scene", "sample", "calibrated_sensor", "sensor"):
            with archive.extractfile(members[name]) as stream:
                tables[name] = json.load(stream)
        lidar_sensors = {row["token"] for row in tables.pop("sensor") if row["channel"] == "LIDAR_TOP"}
        lidar_calibrations = {row["token"] for row in tables["calibrated_sensor"]
                              if row["sensor_token"] in lidar_sensors}
        with archive.extractfile(members["sample_data"]) as stream:
            tables["sample_data"] = [
                row for row in json_items(stream)
                if row["is_key_frame"] and row["calibrated_sensor_token"] in lidar_calibrations
            ]
        needed_poses = {row["ego_pose_token"] for row in tables["sample_data"]}
        with archive.extractfile(members["ego_pose"]) as stream:
            tables["ego_pose"] = [row for row in json_items(stream) if row["token"] in needed_poses]
    if not needed_poses or needed_poses != {row["token"] for row in tables["ego_pose"]}:
        raise ValueError("Missing keyframe LiDAR poses in the nuScenes trainval metadata")
    return {"tables": tables}


def label_index(path):
    source = fingerprint(path)
    members = {}
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if member.isfile() and name.startswith("gts/") and name.endswith("/labels.npz"):
                if name in members or member.size <= 0 or member.offset_data + member.size > source["size"]:
                    raise ValueError(f"Invalid or duplicate label file: {name}")
                members[name] = [str(path), member.offset_data, member.size]
            archive.members.clear()
    if not members:
        raise ValueError("Expected gts/<scene>/<sample>/labels.npz in the Occ3D archive")
    if fingerprint(path) != source:
        raise ValueError("Occ3D archive changed during indexing")
    return {"sources": [source], "members": members}


def write_index(path, index):
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(index, stream, separators=(",", ":"))
    temporary.replace(path)


def main():
    archive_dir = ROOT / "data/nuscenes_archives"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=archive_dir / "v1.0-trainval_meta.tar")
    parser.add_argument("--labels", type=Path, default=archive_dir / "occ3d_labels.tar")
    parser.add_argument("--output", type=Path, default=ROOT / "data/manifests")
    args = parser.parse_args()

    print("Indexing nuScenes LiDAR keyframe metadata", flush=True)
    metadata = metadata_index(args.metadata.resolve(strict=True))
    print("Indexing Occ3D label files", flush=True)
    labels = label_index(args.labels.resolve(strict=True))
    args.output.mkdir(parents=True, exist_ok=True)
    write_index(args.output / "metadata_index.json", metadata)
    write_index(args.output / "label_archive_index.json", labels)
    print(f"Wrote indexes for {len(labels['members'])} label files to {args.output}")


if __name__ == "__main__":
    main()
