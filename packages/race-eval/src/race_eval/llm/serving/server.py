"""OpenAI-compatible FastAPI inference server with RACE weight-surgery support.

Loads one HuggingFace causal LM replica per GPU, applies optional RACE
neuron-intervention **directly to the model weights** (no forward hooks),
and serves requests via a standard ``POST /v1/chat/completions`` endpoint.

Instead of registering ``forward_pre_hook`` callbacks at runtime (which add
Python overhead to every ``model.generate()`` call), this server modifies the
weight matrices in-place before inference begins:

- **suppress / keep_top** — zero out the relevant weight columns.
- **enhance**             — scale the relevant weight columns by the factor.
- **mean_ablation**       — fold the calibration mean into a bias term and
  zero the columns, so inference requires no hooks at all.

Dynamic batching: incoming requests are queued and dispatched in batches
when either ``max_batch_size`` requests have accumulated or
``batch_timeout_ms`` milliseconds have elapsed, whichever comes first.

Multi-GPU data parallelism: each replica runs in its own **process**
(via ``multiprocessing`` with ``spawn`` start-method), eliminating GIL
contention and allowing every GPU to run ``model.generate()`` in true
parallel.

Usage::

    python -m race_eval.llm.serving.server \\
        --model  Qwen/Qwen3-4B-Instruct-2507 \\
        --device-ids 0 1 2 3 \\
        --race-h5  result/llm/online_race/.../online_race.h5 \\
        --concept   math500 \\
        --operation mean_ablation \\
        --top-k 8  --metric lcb_pos --lcb-min-score 0 \\
        --modules   attn \\
        --host 0.0.0.0 --port 8000

    # Baseline (no weight surgery):
    python -m race_eval.llm.serving.server \\
        --model Qwen/Qwen3-4B-Instruct-2507 \\
        --device-ids 0 1 \\
        --no-race
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import time
import uuid
import multiprocessing as mp
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ======================================================================
# Pydantic schemas — OpenAI chat completion wire format
# ======================================================================


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=4096)
    temperature: Optional[float] = Field(default=0.0)
    top_p: Optional[float] = Field(default=1.0)
    top_k: Optional[int] = Field(default=None)
    n: int = Field(default=1)
    stream: bool = Field(default=False)


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "race"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard]


# ======================================================================
# Internal request/response dataclasses
# ======================================================================


@dataclass
class _PendingRequest:
    """A single inference request waiting in the batch queue."""

    messages: List[Dict[str, str]]
    max_tokens: int
    temperature: float
    top_p: float
    top_k: Optional[int]
    future: asyncio.Future  # resolved with the generated text string


# ======================================================================
# GPU Replica — one model per device
# ======================================================================


class _GPUReplica:
    """One model replica pinned to a single GPU.

    Handles prompt building, tokenisation, generation, and decoding for
    a batch of requests.  All calls are made from a dedicated thread so
    the asyncio event loop is never blocked.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str,
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    # ------------------------------------------------------------------
    # Batch inference
    # ------------------------------------------------------------------

    def run_batch(
        self,
        batch: List[Dict[str, Any]],
    ) -> List[str]:
        """Run a batch of requests in a single ``model.generate()`` call.

        Each item is a plain dict with keys ``messages``, ``max_tokens``,
        ``temperature``, ``top_p``, ``top_k`` (all JSON-serialisable so they
        can travel through a multiprocessing queue).
        """
        if not batch:
            return []

        prompts = [self._build_prompt(item["messages"]) for item in batch]
        inputs = self._tokenize_prompts(prompts)
        prompt_len = inputs["input_ids"].shape[1]

        cfg = batch[0]
        gen_kwargs = self._build_gen_kwargs(
            max_tokens=cfg["max_tokens"],
            temperature=cfg["temperature"],
            top_p=cfg["top_p"],
            top_k=cfg.get("top_k"),
        )

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **gen_kwargs)

        gen_ids = output_ids[:, prompt_len:]
        return self._tokenizer.batch_decode(
            gen_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _tokenize_prompts(self, prompts: Sequence[str]) -> Dict[str, torch.Tensor]:
        return self._tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self._device)

    def _build_gen_kwargs(
        self,
        max_tokens: int,
        temperature: float,
        top_p: float,
        top_k: Optional[int],
    ) -> Dict[str, Any]:
        do_sample = temperature > 0.0
        kwargs: Dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "pad_token_id": self._tokenizer.pad_token_id,
            "eos_token_id": self._tokenizer.eos_token_id,
            "do_sample": do_sample,
        }
        if do_sample:
            kwargs["temperature"] = temperature
            if top_p is not None and top_p < 1.0:
                kwargs["top_p"] = top_p
            if top_k is not None:
                kwargs["top_k"] = top_k
        return kwargs


# ======================================================================
# GPU worker process — one per device, independent GIL
# ======================================================================


def _gpu_worker_fn(
    device_id: int,
    args: argparse.Namespace,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    ready_event,
) -> None:
    """Entry-point for a GPU worker process.

    Each worker owns one model replica in its own Python interpreter, so
    ``model.generate()`` runs without GIL contention from other GPUs.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [Worker-%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(f"gpu{device_id}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.cuda.set_device(device_id)
    device = f"cuda:{device_id}"

    from race_eval.llm.evaluation.utils import build_plan, load_model
    from race_eval.llm.pipelines.save_perturbed_model import (
        modify_weights_inplace,
        _patch_config_for_mean_ablation,
    )

    log.info("Loading model on %s ...", device)
    model, tokenizer = load_model(
        args.model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        device=device,
    )

    if not args.no_race:
        plan = build_plan(args)
        mean_activations = None
        if args.operation == "mean_ablation":
            from race_eval.llm.evaluation.mean_activations import (
                compute_mean_activations,
            )

            calibration_corpus = getattr(args, "calibration_corpus", "wikitext2")
            calibration_samples = getattr(args, "calibration_samples", 200)
            log.info(
                "Computing calibration means on %s (%d samples) ...",
                calibration_corpus,
                calibration_samples,
            )
            mean_acts = compute_mean_activations(
                model,
                tokenizer,
                calibration_corpus=calibration_corpus,
                max_samples=calibration_samples,
                max_length=args.max_length,
                batch_size=args.batch_size,
            )
            mean_activations = {k: v.cpu() for k, v in mean_acts.items()}
        stats = modify_weights_inplace(model, plan, mean_activations=mean_activations)
        if args.operation == "mean_ablation":
            _patch_config_for_mean_ablation(model, plan)
        log.info(
            "Applied RACE weight surgery: %d attn + %d mlp neurons modified (%s)",
            stats["attn_neurons_modified"],
            stats["mlp_neurons_modified"],
            args.operation,
        )

    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        log.warning("torch.compile is unavailable; running in eager mode")
    else:
        try:
            model = compile_fn(model, mode="max-autotune")
            log.info("Compiled model with torch.compile")
        except Exception:
            log.exception("torch.compile failed; falling back to eager mode")

    replica = _GPUReplica(model, tokenizer, device)
    ready_event.set()
    log.info("Ready on %s", device)

    while True:
        try:
            batch_data = in_queue.get()
        except (EOFError, KeyboardInterrupt):
            break
        if batch_data is None:  # shutdown sentinel
            break
        try:
            log.info(
                "Starting inference on %s with batch_size=%d", device, len(batch_data)
            )
            texts = replica.run_batch(batch_data)
            out_queue.put(texts)
        except Exception as exc:
            log.exception("Batch inference error")
            out_queue.put(RuntimeError(str(exc)))


class _GPUWorkerHandle:
    """Main-process handle for a GPU worker living in a separate process."""

    def __init__(self, device_id: int, args: argparse.Namespace) -> None:
        ctx = mp.get_context("spawn")
        self.device_id = device_id
        self._in_q: mp.Queue = ctx.Queue()
        self._out_q: mp.Queue = ctx.Queue()
        self._ready = ctx.Event()
        self._proc = ctx.Process(
            target=_gpu_worker_fn,
            args=(device_id, args, self._in_q, self._out_q, self._ready),
            daemon=True,
        )

    def start(self) -> None:
        self._proc.start()

    def wait_ready(self, timeout: float = 600.0) -> None:
        if not self._ready.wait(timeout=timeout):
            raise RuntimeError(
                f"GPU worker {self.device_id} not ready within {timeout}s"
            )

    def submit(self, items: List[Dict[str, Any]]) -> None:
        """Send a serialisable sub-batch to the worker."""
        self._in_q.put(items)

    def collect(self) -> List[str]:
        """Block until the worker returns results (or an exception)."""
        return self._out_q.get()

    def stop(self) -> None:
        self._in_q.put(None)
        self._proc.join(timeout=30)
        if self._proc.is_alive():
            self._proc.kill()


# ======================================================================
# Dynamic batcher + multi-GPU dispatcher
# ======================================================================


def _log_task_exception(task: asyncio.Task) -> None:
    """Callback attached to fire-and-forget tasks to surface unhandled errors."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Unhandled exception in dispatch task", exc_info=exc)


class RaceInferenceServer:
    """Core server logic: dynamic batching + multi-GPU dispatch via worker processes.

    Each GPU worker is a **separate process** with its own GIL, so generation
    on every GPU proceeds in true parallel without Python-level contention.

    Workflow:
        1. Collect up to ``max_batch_size`` requests (or wait ``batch_timeout_ms``).
        2. Split the batch into N equal chunks — one chunk per GPU worker.
        3. Send each chunk to the corresponding worker process via a
           multiprocessing queue.
        4. Gather all results and resolve the per-request futures.

    Args:
        workers: One :class:`_GPUWorkerHandle` per GPU.
        model_name: Logical model name advertised on ``/v1/models``.
        max_batch_size: Total requests accumulated before a dispatch is triggered.
        batch_timeout_ms: Maximum wait time (ms) before dispatching an incomplete
            batch (whichever comes first: full batch or timeout).
    """

    def __init__(
        self,
        workers: List[_GPUWorkerHandle],
        model_name: str,
        max_batch_size: int = 32,
        batch_timeout_ms: float = 50.0,
    ) -> None:
        if not workers:
            raise ValueError("At least one GPU worker is required")
        self._workers = workers
        self.model_name = model_name
        self._max_batch_size = max_batch_size
        self._batch_timeout_s = batch_timeout_ms / 1000.0
        self._queue: asyncio.Queue[_PendingRequest] = asyncio.Queue()
        self._batcher_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background batching loop."""
        self._batcher_task = asyncio.create_task(self._batcher_loop())
        logger.info(
            "RaceInferenceServer started: %d GPU(s), max_batch=%d, timeout=%.0fms",
            len(self._workers),
            self._max_batch_size,
            self._batch_timeout_s * 1000,
        )

    async def stop(self) -> None:
        """Cancel the batching loop and stop all GPU worker processes."""
        if self._batcher_task is not None:
            self._batcher_task.cancel()
            try:
                await self._batcher_task
            except asyncio.CancelledError:
                pass
        for w in self._workers:
            w.stop()
        logger.info("RaceInferenceServer stopped.")

    # ------------------------------------------------------------------
    # Public inference entry point
    # ------------------------------------------------------------------

    async def chat_complete(self, request: ChatCompletionRequest) -> str:
        """Enqueue a chat completion request and await its result."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        pending = _PendingRequest(
            messages=[m.model_dump() for m in request.messages],
            max_tokens=request.max_tokens or 4096,
            temperature=request.temperature if request.temperature is not None else 0.0,
            top_p=request.top_p if request.top_p is not None else 1.0,
            top_k=request.top_k,
            future=future,
        )
        await self._queue.put(pending)
        return await future

    # ------------------------------------------------------------------
    # Batching loop
    # ------------------------------------------------------------------

    async def _batcher_loop(self) -> None:
        """Collect requests then dispatch each full batch to all GPUs."""
        while True:
            batch: List[_PendingRequest] = []
            deadline = asyncio.get_event_loop().time() + self._batch_timeout_s

            # Block until the first request arrives
            try:
                first = await self._queue.get()
                batch.append(first)
            except asyncio.CancelledError:
                return

            # Collect more until max_batch_size or timeout
            while len(batch) < self._max_batch_size:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
                except asyncio.CancelledError:
                    for req in batch:
                        if not req.future.done():
                            req.future.cancel()
                    return

            if batch:
                # Fire-and-forget: dispatch does not block the batcher loop.
                # add_done_callback ensures unhandled exceptions are logged.
                task = asyncio.create_task(self._dispatch_batch(batch))
                task.add_done_callback(_log_task_exception)

    # ------------------------------------------------------------------
    # Data-parallel dispatch
    # ------------------------------------------------------------------

    async def _dispatch_batch(self, batch: List[_PendingRequest]) -> None:
        """Split a batch across GPU worker processes and collect in parallel.

        Each chunk is sent via a multiprocessing queue, and results are
        collected concurrently using ``asyncio.gather`` (the blocking
        ``queue.get()`` is offloaded to the default thread executor so the
        event loop is never blocked).
        """
        n = len(self._workers)
        total = len(batch)
        chunk_indices: List[List[int]] = [list(range(i, total, n)) for i in range(n)]

        loop = asyncio.get_running_loop()

        # Submit serialisable sub-batches to each worker
        active: List[Tuple[List[int], _GPUWorkerHandle]] = []
        for i, worker in enumerate(self._workers):
            indices = chunk_indices[i]
            if not indices:
                continue
            items = [
                {
                    "messages": batch[j].messages,
                    "max_tokens": batch[j].max_tokens,
                    "temperature": batch[j].temperature,
                    "top_p": batch[j].top_p,
                    "top_k": batch[j].top_k,
                }
                for j in indices
            ]
            worker.submit(items)
            active.append((indices, worker))

        # Collect results from all workers concurrently
        async def _collect(
            indices: List[int],
            w: _GPUWorkerHandle,
        ) -> List[Tuple[int, str]]:
            result = await loop.run_in_executor(None, w.collect)
            if isinstance(result, Exception):
                raise result
            return list(zip(indices, result))

        results: List[str] = [""] * total
        try:
            gathered = await asyncio.gather(
                *[_collect(idx, w) for idx, w in active],
            )
            for pairs in gathered:
                for orig_idx, text in pairs:
                    results[orig_idx] = text
        except Exception as exc:
            logger.exception("Batch inference failed")
            for req in batch:
                if not req.future.done():
                    req.future.set_exception(exc)
            return

        for req, text in zip(batch, results):
            if not req.future.done():
                req.future.set_result(text)


# ======================================================================
# FastAPI application factory
# ======================================================================


def create_app(server: RaceInferenceServer) -> FastAPI:
    """Create and configure the FastAPI application."""

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):  # noqa: ARG001
        await server.start()
        yield
        await server.stop()

    app = FastAPI(title="RACE Inference Server", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok", "model": server.model_name}

    @app.get("/v1/models", response_model=ModelList)
    async def list_models():
        return ModelList(data=[ModelCard(id=server.model_name)])

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    async def chat_completions(request: ChatCompletionRequest):
        t0 = time.time()
        print(f"chat_completions request")
        try:
            text = await server.chat_complete(request)
        except Exception as exc:
            logger.exception("Inference error")
            return JSONResponse(
                status_code=500,
                content={"error": {"message": str(exc), "type": "inference_error"}},
            )

        prompt_tokens = sum(len(m.content.split()) for m in request.messages)
        completion_tokens = len(text.split())

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
            created=int(t0),
            model=server.model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason="stop",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    return app


# ======================================================================
# Server bootstrap — model loading + hook setup
# ======================================================================


def _launch_workers(
    args: argparse.Namespace,
) -> Tuple[List[_GPUWorkerHandle], str]:
    """Spawn one worker process per GPU and wait for readiness.

    Returns ``(workers, model_name)``.
    """
    device_ids: List[int] = args.device_ids
    logger.info("Spawning %d GPU worker processes ...", len(device_ids))

    workers: List[_GPUWorkerHandle] = []
    for dev_id in device_ids:
        w = _GPUWorkerHandle(dev_id, args)
        w.start()
        workers.append(w)

    for w in workers:
        w.wait_ready(timeout=600.0)
    logger.info("All %d GPU workers ready.", len(workers))

    use_race = not args.no_race
    suffix = f"__{args.operation}" if use_race else "__baseline"
    model_name = args.served_model_name or (args.model.replace("/", "--") + suffix)
    return workers, model_name


# ======================================================================
# CLI
# ======================================================================


def _add_args(parser: argparse.ArgumentParser) -> None:
    # --- Model ---
    parser.add_argument(
        "--model", required=True, help="HuggingFace model identifier or local path"
    )
    parser.add_argument(
        "--dtype",
        "--torch-dtype",
        dest="dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="Override the model name advertised by /v1/models",
    )

    # --- GPU ---
    parser.add_argument(
        "--device-ids",
        nargs="+",
        type=int,
        default=[0],
        metavar="ID",
        help="CUDA device IDs for data-parallel replicas",
    )

    # --- RACE weight surgery (optional) ---
    parser.add_argument(
        "--no-race",
        action="store_true",
        help="Disable RACE weight surgery (run pure baseline)",
    )
    parser.add_argument("--race-h5", default=None, help="Path to RACE results H5 file")
    parser.add_argument(
        "--concept", default="code_generation", help="Concept name in RACE results"
    )
    parser.add_argument(
        "--operation",
        default="mean_ablation",
        choices=["suppress", "enhance", "keep_top", "mean_ablation"],
    )
    parser.add_argument("--top-k-percent", type=float, default=5.0)
    parser.add_argument(
        "--top-k", type=int, default=None, dest="top_k_count", metavar="N"
    )
    parser.add_argument(
        "--metric",
        default="posterior_mean",
        choices=[
            "posterior_mean",
            "lcb",
            "lcb_pos",
            "lcb_neg",
            "empirical_mean",
            "empirical_snr",
            "activation_mean",
        ],
    )
    parser.add_argument("--lcb-min-score", type=float, default=None, metavar="T")
    parser.add_argument(
        "--modules", nargs="+", default=["attn", "mlp"], choices=["attn", "mlp"]
    )
    parser.add_argument("--enhancement-factor", type=float, default=2.0)

    # --- Mean ablation calibration ---
    parser.add_argument(
        "--calibration-corpus",
        default="wikitext2",
        help="Corpus for computing calibration mean activations (mean_ablation only)",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=200,
        help="Number of calibration passages for mean activations (mean_ablation only)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2048,
        help="Max token length per calibration passage (mean_ablation only)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for calibration forward passes (mean_ablation only)",
    )

    # --- Server ---
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Maximum requests per inference batch",
    )
    parser.add_argument(
        "--batch-timeout-ms",
        type=float,
        default=50.0,
        help="Max wait time (ms) before dispatching an incomplete batch",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn worker processes (keep 1 for GPU sharing)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RACE OpenAI-compatible inference server",
    )
    _add_args(parser)
    args = parser.parse_args()

    if not args.no_race and args.race_h5 is None:
        parser.error("--race-h5 is required unless --no-race is set")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    logger.info("=" * 60)
    logger.info("  RACE Inference Server")
    logger.info("=" * 60)
    logger.info("  model=%s  device_ids=%s", args.model, args.device_ids)
    if not args.no_race:
        logger.info(
            "  RACE weight surgery: concept=%s  op=%s  top_k=%s",
            args.concept,
            args.operation,
            args.top_k_count or f"{args.top_k_percent}%",
        )
    else:
        logger.info("  RACE weight surgery: DISABLED (baseline mode)")
    logger.info(
        "  listen=%s:%d  max_batch=%d  timeout=%.0fms",
        args.host,
        args.port,
        args.max_batch_size,
        args.batch_timeout_ms,
    )
    logger.info("=" * 60)

    workers, model_name = _launch_workers(args)
    logger.info("Model name advertised: %s", model_name)

    server = RaceInferenceServer(
        workers=workers,
        model_name=model_name,
        max_batch_size=args.max_batch_size,
        batch_timeout_ms=args.batch_timeout_ms,
    )
    app = create_app(server)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
