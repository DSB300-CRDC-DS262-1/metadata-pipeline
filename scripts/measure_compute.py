"""Measure GPU memory, power and energy while local models run metadata extraction.

Usage: python scripts/measure_compute.py phi4 gemma4:e4b gemma4:12b:off
       (append ":off" to run a model with thinking off)
"""

from __future__ import annotations

import glob
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
RUNS = 3
QUERY = ["nvidia-smi", "--query-gpu=power.draw,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]


def gpu_sample() -> tuple[float, float, float]:
    power, memory, utilisation = subprocess.run(QUERY, capture_output=True, text=True).stdout.split(",")
    return float(power), float(memory), float(utilisation)


def idle_power(seconds: float = 5) -> float:
    samples = []
    for _ in range(int(seconds / 0.5)):
        samples.append(gpu_sample()[0])
        time.sleep(0.5)
    return sum(samples) / len(samples)


def measure(model: str, think: str, suffix: str) -> dict:
    samples: list[tuple[float, float, float, float]] = []
    done = threading.Event()

    def sampler() -> None:
        while not done.is_set():
            samples.append((time.monotonic(), *gpu_sample()))
            time.sleep(0.5)

    thread = threading.Thread(target=sampler)
    thread.start()
    start = time.monotonic()
    command = [PYTHON, "scripts/run_metadata_extraction.py", "--model", model, "--think", think]
    if suffix:
        command += ["--suffix", suffix]
    output = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8").stdout
    split = subprocess.run(["ollama", "ps"], capture_output=True, text=True).stdout.splitlines()[-1]
    done.set()
    thread.join()

    energy = sum(p * (t2 - t1) for (t1, p, *_), (t2, *_) in zip(samples, samples[1:]))
    score = next((line.split(": ")[1].split(" reference")[0] for line in output.splitlines()
                  if "reference fields matched" in line), "failed")
    return {
        "seconds": time.monotonic() - start,
        "energy_j": energy,
        "avg_power_w": energy / max(samples[-1][0] - samples[0][0], 1e-9),
        "peak_memory_mb": max(s[2] for s in samples),
        "avg_utilisation": sum(s[3] for s in samples) / len(samples),
        "processor": " ".join(split.split()[3:5]) if "%" in split else "?",
        "score": score,
    }


def tokens(label: str) -> tuple[float, float]:
    docs = [json.load(open(f, encoding="utf-8")) for f in glob.glob(str(PROJECT_ROOT / f"output/metadata/{label}*/*.json"))]
    return (sum(d["prompt_tokens"] or 0 for d in docs) / len(docs), sum(d["output_tokens"] or 0 for d in docs) / len(docs))


def main() -> None:
    idle = idle_power()
    print(f"idle GPU power {idle:.0f} W\n")
    print(f"{'model':<20}{'score':<9}{'s/doc':<8}{'Wh/doc':<9}{'above idle':<12}{'peak VRAM':<11}{'GPU use':<9}{'split':<14}{'tokens in/out'}")
    for spec in sys.argv[1:]:
        model, _, think = spec.partition(":off") if spec.endswith(":off") else (spec, "", "")
        think = "off" if spec.endswith(":off") else "default"
        subprocess.run(["ollama", "stop", model], capture_output=True)
        results = [measure(model, think, suffix) for suffix in ["", "r2", "r3"][:RUNS]]
        warm = results[1:] or results  # first run includes loading the model
        per_doc = lambda key: sum(r[key] for r in warm) / len(warm) / 4  # noqa: E731
        label = "ollama_" + model.replace(":", "-") + ("_nothink" if think == "off" else "")
        tin, tout = tokens(label)
        print(f"{spec:<20}{'/'.join(r['score'].split('/')[0] for r in results):<9}"
              f"{per_doc('seconds'):<8.1f}{per_doc('energy_j') / 3600:<9.2f}"
              f"{(per_doc('energy_j') - idle * per_doc('seconds')) / 3600:<12.2f}"
              f"{max(r['peak_memory_mb'] for r in results) / 1024:<11.1f}"
              f"{sum(r['avg_utilisation'] for r in warm) / len(warm):<9.0f}"
              f"{warm[-1]['processor']:<14}{tin:.0f}/{tout:.0f}")
        print(f"{'':20}first run (includes loading): {results[0]['seconds']:.0f}s")


if __name__ == "__main__":
    main()
