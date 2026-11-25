"""Read-only access to indexed Occ3D label archives."""
import json
from pathlib import Path


class IndexedArchive:
    def __init__(self, index_path):
        document = json.loads(Path(index_path).read_text())
        for source in document["sources"]:
            path = Path(source["path"])
            stat = path.stat()
            if stat.st_size != source["size"] or stat.st_mtime_ns != source["mtime_ns"]:
                raise RuntimeError(f"Archive changed; rebuild its index: {path}")
        self.members = document["members"]

    def read(self, member):
        path, offset, size = self.members[member]
        with open(path, "rb") as stream:
            stream.seek(offset)
            payload = stream.read(size)
        if len(payload) != size:
            raise IOError(f"Truncated archive member: {member}")
        return payload
