import hashlib, json, os, uuid
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")


def verify_manifest(root, name="MANIFEST.json"):
    root = Path(root)
    m = read(root / name)
    for relative, h in m["files"].items():
        if sha(root / relative) != h:
            raise ValueError("Frozen file changed: " + relative)
    return m


PACKAGE_ROOT = Path(__file__).resolve().parent


def configuration(overrides=None):
    conf = read(PACKAGE_ROOT / "config.json")
    conf.update(overrides or {})
    if conf["coarse_top_k"] != 1 or conf["max_injected_per_checkpoint"] != 1:
        raise ValueError("MUSE requires Top1 per route and at most one injected source")
    if conf["recent_assistant_turns"] != 1:
        raise ValueError("Retrieval requires the current interaction only")
    return conf
