"""Evaluate one or more local models through EvalScope and vLLM.

Each entry in ``MODELS_ROOTS`` or ``--models-roots`` may be a model path or a
directory whose immediate subdirectories are models. Entries can use both
forms. The script starts a vLLM OpenAI-compatible server, evaluates one model,
stops the server, and then continues with the next model.

Set the defaults below or override them on the command line. For example:

python scripts/eval_full_list.py \
  --models-roots /path/to/model_or_models_root \
  --datasets mmlu_redux gpqa_diamond arc \
  --work-dir outputs/mmlu_redux_knowledge_qa
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

from evalscope import TaskConfig, run_task

# ---------------------------------------------------------------------------
# Configurable defaults
# ---------------------------------------------------------------------------

# Each entry may be a model path or a directory containing model directories.
MODELS_ROOTS: List[str] = []
# EvalScope output directory. Reuse it as ``use_cache`` to resume a run.
EVAL_WORK_DIR = "outputs/mmlu_redux_knowledge_qa"
# Override the resume cache with this environment variable.
EVAL_USE_CACHE: Optional[str] = os.environ.get("EVAL_USE_CACHE")
# vLLM readiness polling.
VLLM_READY_TIMEOUT = 3600.0
VLLM_READY_INTERVAL = 2.0

# Grace period between SIGTERM and SIGKILL.
VLLM_STOP_GRACE_SEC = 30.0

# Port used by each sequential vLLM server.
VLLM_PORT = 8001

# Set a CUDA device list such as ``"0"`` or ``"0,1,2,3"``. ``None``
# preserves the current environment.
CUDA_VISIBLE_DEVICES: Optional[str] = None

# Extra arguments passed to ``vllm.entrypoints.openai.api_server``.
VLLM_EXTRA_ARGS: List[str] = [
    "--max-model-len",
    "12k",
    "--gpu-memory-utilization",
    "0.85",
]
# The model ID combines the normalized directory name and this suffix.
MODEL_ID_SUFFIX = ""

EVAL_DATASETS: List[str] = [
    "mmlu_redux",
    "gpqa_diamond",
    "arc",
]
EVAL_API_KEY = "EMPTY"
EVAL_BATCH_SIZE = 32
EVAL_REPEATS = 3
EVAL_SEED = 42
EVAL_GENERATION_CONFIG: Dict = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
}
MBPP_PLUS_ASSISTANT_PREFILL = "```python\n"
MBPP_PLUS_PREFILL_EXTRA_BODY: Dict = {
    "continue_final_message": True,
    "add_generation_prompt": False,
}
HUMANEVAL_INSTRUCTION_PREFIX = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)
HUMANEVAL_PROMPT_TEMPLATE = f"{HUMANEVAL_INSTRUCTION_PREFIX}\n" "{question}"

USE_CUSTOM_HUMANEVAL_PROMPT = False
USE_SANDBOX = True
SANDBOX_TYPE = "docker"
SANDBOX_DATASETS = frozenset({"mbpp_plus", "humaneval", "live_code_bench"})
JUDGE_WORKER_NUM = 5


class ModelDir(NamedTuple):
    name: str
    abs_path: str


def is_model_dir(path: Path) -> bool:
    """Return whether *path* looks like a loadable local model directory."""
    if not path.is_dir():
        return False
    if (path / "config.json").is_file():
        return True
    return any(path.glob("*.safetensors")) or any(path.glob("pytorch_model*.bin"))


def list_subdirs(root: str) -> List[ModelDir]:
    """Return names and absolute paths for immediate child directories."""
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"Not a valid directory: {root_path}")
    out: List[ModelDir] = []
    for p in sorted(root_path.iterdir()):
        if p.is_dir() and not p.name.startswith("."):
            out.append(ModelDir(name=p.name, abs_path=str(p.resolve())))
    return out


def resolve_model_dirs(paths: List[str]) -> List[ModelDir]:
    """Resolve paths into an ordered, deduplicated list of models.

    A directory containing model configuration or weights is one model. Other
    directories expand to their visible immediate subdirectories. A file is
    treated as a single model path.
    """
    result: List[ModelDir] = []
    seen: set[str] = set()
    for p in paths:
        resolved = Path(p).resolve()
        if not resolved.exists():
            print(
                f"Warning: path does not exist; skipping: {resolved}",
                file=sys.stderr,
            )
            continue
        candidates: List[ModelDir]
        if is_model_dir(resolved):
            candidates = [ModelDir(name=resolved.name, abs_path=str(resolved))]
        elif resolved.is_dir():
            subdirs = [
                x
                for x in resolved.iterdir()
                if x.is_dir() and not x.name.startswith(".")
            ]
            if subdirs:
                candidates = list_subdirs(p)
            else:
                candidates = [ModelDir(name=resolved.name, abs_path=str(resolved))]
        else:
            candidates = [ModelDir(name=resolved.stem, abs_path=str(resolved))]

        for candidate in candidates:
            if candidate.abs_path not in seen:
                result.append(candidate)
                seen.add(candidate.abs_path)
    return result


def wait_openai_base_ready(base_url: str, timeout: float, interval: float) -> None:
    """Poll the OpenAI-compatible models endpoint until vLLM is ready."""
    models_url = base_url.rstrip("/") + "/models"
    deadline = time.monotonic() + timeout
    last_err = None
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(models_url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if 200 <= resp.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_err = e
        time.sleep(interval)
    raise RuntimeError(
        f"vLLM was not ready within {timeout}s at {models_url}; "
        f"last error: {last_err}"
    )


def build_vllm_cmd(
    model_path: str,
    port: int,
    served_model_name: str,
    extra_args: List[str],
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_path,
        "--port",
        str(port),
        "--served-model-name",
        served_model_name,
    ]
    cmd.extend(extra_args)
    return cmd


def start_vllm_server(
    cmd: List[str], cuda_visible_devices: Optional[str]
) -> subprocess.Popen:
    # Use a process group so the server and its children stop together. Keep
    # stdout attached to avoid blocking on an unread pipe.
    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    return subprocess.Popen(
        cmd,
        start_new_session=True,
        env=env,
    )


def stop_vllm_server(proc: subprocess.Popen | None, grace: float) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=5)


def make_task_config_for_model(
    served_model_name: str,
    model_id: str,
    api_base: str,
    datasets: List[str],
    generation_config: Dict,
    dataset_args: Optional[Dict] = None,
    *,
    work_dir: str,
    use_cache: Optional[str],
    eval_batch_size: int,
    limit: Optional[int],
    repeats: int,
    seed: int,
) -> TaskConfig:
    use_sandbox = USE_SANDBOX and any(
        dataset in SANDBOX_DATASETS for dataset in datasets
    )
    return TaskConfig(
        model=served_model_name,
        model_id=model_id,
        datasets=datasets,
        api_url=api_base,
        api_key=EVAL_API_KEY,
        eval_type="openai_api",
        eval_batch_size=eval_batch_size,
        generation_config=generation_config,
        dataset_args=dataset_args or {},
        use_sandbox=use_sandbox,
        sandbox_type=SANDBOX_TYPE,
        judge_worker_num=JUDGE_WORKER_NUM,
        use_cache=use_cache,
        work_dir=work_dir,
        no_timestamp=True,
        limit=limit,
        repeats=repeats,
        seed=seed,
    )


def make_mbpp_plus_prefill_generation_config() -> Dict:
    """Build generation config for MBPP+ assistant-side prefill.

    EvalScope's generation_config is task-wide, so mbpp_plus must be evaluated
    separately from datasets whose final message is not an assistant prefill.
    """
    config = deepcopy(EVAL_GENERATION_CONFIG)
    extra_body = deepcopy(config.get("extra_body", {}))
    extra_body.update(MBPP_PLUS_PREFILL_EXTRA_BODY)
    config["extra_body"] = extra_body
    return config


def build_eval_task_specs(
    datasets: List[str],
) -> List[Tuple[str, List[str], Dict, Dict]]:
    specs: List[Tuple[str, List[str], Dict, Dict]] = []
    mbpp_plus_enabled = "mbpp_plus" in datasets
    other_datasets = [dataset for dataset in datasets if dataset != "mbpp_plus"]

    if mbpp_plus_enabled:
        specs.append(
            (
                "mbpp_plus_prefill",
                ["mbpp_plus"],
                make_mbpp_plus_prefill_generation_config(),
                {
                    "mbpp_plus": {
                        "extra_params": {
                            "assistant_prefill": MBPP_PLUS_ASSISTANT_PREFILL,
                        }
                    }
                },
            )
        )

    if other_datasets:
        dataset_args = {}
        if USE_CUSTOM_HUMANEVAL_PROMPT and "humaneval" in other_datasets:
            dataset_args["humaneval"] = {
                "prompt_template": HUMANEVAL_PROMPT_TEMPLATE,
            }
        specs.append(
            (
                "other_datasets",
                other_datasets,
                deepcopy(EVAL_GENERATION_CONFIG),
                dataset_args,
            )
        )

    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate local models sequentially with vLLM and EvalScope."
    )
    parser.add_argument(
        "--models-roots",
        nargs="+",
        default=MODELS_ROOTS,
        help="Model paths, directories containing models, or a mixture of both.",
    )
    parser.add_argument("--datasets", nargs="+", default=EVAL_DATASETS)
    parser.add_argument("--work-dir", default=EVAL_WORK_DIR)
    parser.add_argument(
        "--use-cache",
        default=EVAL_USE_CACHE,
        help="EvalScope cache directory. Defaults to --work-dir.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable the EvalScope resume cache.",
    )
    parser.add_argument("--cuda-visible-devices", default=CUDA_VISIBLE_DEVICES)
    parser.add_argument("--port", type=int, default=VLLM_PORT)
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument(
        "--repeats",
        type=int,
        default=EVAL_REPEATS,
        help="Evaluation repeats per benchmark sample. Defaults to 3.",
    )
    parser.add_argument("--seed", type=int, default=EVAL_SEED)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum examples per dataset subset. Intended for smoke tests.",
    )
    parser.add_argument("--model-id-suffix", default=MODEL_ID_SUFFIX)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved models, tasks, and vLLM commands without running them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.models_roots:
        print("Set --models-roots or MODELS_ROOTS", file=sys.stderr)
        sys.exit(1)
    if args.repeats < 1:
        print("--repeats must be at least 1", file=sys.stderr)
        sys.exit(1)

    model_dirs = resolve_model_dirs(args.models_roots)
    if not model_dirs:
        print(
            "No model paths found; check the MODELS_ROOTS configuration",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {len(model_dirs)} model paths to evaluate sequentially.")
    for md in model_dirs:
        print(f"  - {md.name}: {md.abs_path}")

    task_specs = build_eval_task_specs(args.datasets)
    if not task_specs:
        print("No evaluation datasets configured", file=sys.stderr)
        sys.exit(1)

    use_cache = None if args.no_cache else (args.use_cache or args.work_dir)

    for task_name, datasets, _, _ in task_specs:
        print(f"  - Task {task_name}: {datasets}")

    for md in model_dirs:
        port = args.port
        api_base = f"http://127.0.0.1:{port}/v1"
        served_name = md.name.replace(" ", "_")
        model_id = f"{served_name}{args.model_id_suffix}"
        cmd = build_vllm_cmd(md.abs_path, port, served_name, VLLM_EXTRA_ARGS)

        print(f"\n======== Model: {md.name} | Port: {port} ========")
        print("Starting vLLM:", " ".join(cmd))
        if args.cuda_visible_devices is not None:
            print(f"CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")
        print(f"EvalScope work_dir={args.work_dir} use_cache={use_cache}")
        print(f"EvalScope repeats={args.repeats} seed={args.seed}")
        if args.dry_run:
            continue

        proc: subprocess.Popen | None = None
        try:
            proc = start_vllm_server(cmd, args.cuda_visible_devices)
            wait_openai_base_ready(
                api_base,
                timeout=VLLM_READY_TIMEOUT,
                interval=VLLM_READY_INTERVAL,
            )
            for task_name, datasets, generation_config, dataset_args in task_specs:
                print(f"Running EvalScope task: {task_name} datasets={datasets}")
                if dataset_args:
                    print(f"dataset_args={dataset_args}")
                task_cfg = make_task_config_for_model(
                    served_name,
                    model_id,
                    api_base,
                    datasets=datasets,
                    generation_config=generation_config,
                    dataset_args=dataset_args,
                    work_dir=args.work_dir,
                    use_cache=use_cache,
                    eval_batch_size=args.eval_batch_size,
                    limit=args.limit,
                    repeats=args.repeats,
                    seed=args.seed,
                )
                run_task(task_cfg)
            print(f"Evaluation completed: {md.name}")
        except Exception as e:
            print(f"Model {md.name} failed: {e}", file=sys.stderr)
            raise
        finally:
            print(f"Stopping vLLM: {md.name}")
            stop_vllm_server(proc, grace=VLLM_STOP_GRACE_SEC)


if __name__ == "__main__":
    main()
