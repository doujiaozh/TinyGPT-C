import json
import os
import struct
import sys
import time

import numpy as np
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
BASE = "https://hf-mirror.com/distilgpt2/resolve/main"
SMALL = {
    "config.json": 600,
    "vocab.json": 900000,
    "merges.txt": 400000,
}
WEIGHTS = "model.safetensors"
WEIGHTS_MIN = 300000000


def fetch(name, minimum):
    path = os.path.join(ASSETS, name)
    if os.path.exists(path) and os.path.getsize(path) >= minimum:
        print("asset present", name, os.path.getsize(path), flush=True)
        return path
    os.makedirs(ASSETS, exist_ok=True)
    part = path + ".part"
    pos = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {"User-Agent": "tinygpt-c/1.0"}
    if pos:
        headers["Range"] = "bytes={}-".format(pos)
    url = BASE + "/" + name
    print("download start", name, "offset", pos, flush=True)
    with requests.get(url, headers=headers, stream=True, timeout=(20, 120), allow_redirects=True) as response:
        if response.status_code == 416 and pos:
            os.replace(part, path)
            return path
        response.raise_for_status()
        append = pos > 0 and response.status_code == 206
        mode = "ab" if append else "wb"
        if not append:
            pos = 0
        done = pos
        last = time.time()
        with open(part, mode) as handle:
            for chunk in response.iter_content(1048576):
                if not chunk:
                    continue
                handle.write(chunk)
                done += len(chunk)
                if time.time() - last >= 3.0:
                    print("download", name, "{:.1f} MiB".format(done / 1048576.0), flush=True)
                    last = time.time()
    actual = os.path.getsize(part)
    if actual < minimum:
        raise RuntimeError("incomplete download {} {}".format(name, actual))
    os.replace(part, path)
    print("download complete", name, actual, flush=True)
    return path


def tensor_names(config):
    names = ["wte.weight", "wpe.weight", "ln_f.weight", "ln_f.bias"]
    for layer in range(config["n_layer"]):
        prefix = "h.{}.ln_1".format(layer)
        names.extend([prefix + ".weight", prefix + ".bias"])
        prefix = "h.{}.attn.c_attn".format(layer)
        names.extend([prefix + ".weight", prefix + ".bias"])
        prefix = "h.{}.attn.c_proj".format(layer)
        names.extend([prefix + ".weight", prefix + ".bias"])
        prefix = "h.{}.ln_2".format(layer)
        names.extend([prefix + ".weight", prefix + ".bias"])
        prefix = "h.{}.mlp.c_fc".format(layer)
        names.extend([prefix + ".weight", prefix + ".bias"])
        prefix = "h.{}.mlp.c_proj".format(layer)
        names.extend([prefix + ".weight", prefix + ".bias"])
    return names


def convert(weights_path, config_path, output_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        raw_config = json.load(handle)
    config = {
        "n_layer": int(raw_config["n_layer"]),
        "n_head": int(raw_config["n_head"]),
        "n_embd": int(raw_config["n_embd"]),
        "n_ctx": int(raw_config["n_ctx"]),
        "vocab_size": int(raw_config["vocab_size"]),
        "n_inner": int(raw_config.get("n_inner") or 4 * int(raw_config["n_embd"])),
    }
    with open(weights_path, "rb") as handle:
        header_length = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_length).decode("utf-8"))
        data_start = 8 + header_length
    names = tensor_names(config)
    available = {}
    for key, value in header.items():
        if key == "__metadata__":
            continue
        canonical = key[len("transformer."):] if key.startswith("transformer.") else key
        available[canonical] = value
    missing = [name for name in names if name not in available]
    if missing:
        raise RuntimeError("missing tensors: {}".format(", ".join(missing[:8])))
    dtype_map = {
        "F32": np.dtype("<f4"),
        "F16": np.dtype("<f2"),
    }
    total = 0
    temporary = output_path + ".part"
    with open(weights_path, "rb") as source, open(temporary, "wb") as target:
        target.write(b"TGPTBIN1")
        for value in (1, config["n_layer"], config["n_head"], config["n_embd"], config["n_ctx"], config["vocab_size"], config["n_inner"], len(names)):
            target.write(struct.pack("<I", value))
        for index, name in enumerate(names):
            info = available[name]
            dtype = info["dtype"]
            shape = tuple(int(value) for value in info["shape"])
            start, end = (int(value) for value in info["data_offsets"])
            source.seek(data_start + start)
            raw = source.read(end - start)
            if dtype in dtype_map:
                array = np.frombuffer(raw, dtype=dtype_map[dtype]).astype(np.float32, copy=False)
            elif dtype == "BF16":
                array = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
            else:
                raise RuntimeError("unsupported dtype {} for {}".format(dtype, name))
            array = np.ascontiguousarray(array.reshape(shape), dtype="<f4")
            expected = 1
            for value in shape:
                expected *= value
            if expected != array.size:
                raise RuntimeError("shape mismatch {}".format(name))
            name_bytes = name.encode("utf-8")
            target.write(struct.pack("<I", len(name_bytes)))
            target.write(name_bytes)
            target.write(struct.pack("<I", len(shape)))
            for value in shape:
                target.write(struct.pack("<Q", value))
            target.write(struct.pack("<I", 0))
            target.write(struct.pack("<Q", array.nbytes))
            target.write(array.tobytes(order="C"))
            total += array.size
            print("packed", index + 1, len(names), name, shape, flush=True)
    os.replace(temporary, output_path)
    print("model written", output_path, os.path.getsize(output_path), flush=True)
    print("parameters", total, flush=True)


def main():
    os.makedirs(ASSETS, exist_ok=True)
    for name, minimum in SMALL.items():
        fetch(name, minimum)
    weights = fetch(WEIGHTS, WEIGHTS_MIN)
    config_path = os.path.join(ASSETS, "config.json")
    output_path = os.path.join(ASSETS, "model.bin")
    convert(weights, config_path, output_path)


if __name__ == "__main__":
    main()
