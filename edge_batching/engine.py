from __future__ import annotations

from dataclasses import dataclass, field
import random
import time
from typing import Callable, Protocol, Sequence

from .models import BatchExecutionMetrics, GenerationRequest, GenerationResult


class BatchEngine(Protocol):
    def execute_batch(
        self,
        batch: Sequence[GenerationRequest],
    ) -> tuple[list[GenerationResult], BatchExecutionMetrics]:
        """Execute a request batch and return individual outputs + runtime metrics."""


@dataclass(slots=True)
class MockLlamaCppEngine:
    """
    Timing model for local scheduler validation.

    This does not run a model. It emulates latency behavior where batching
    increases throughput but can increase single-request tail latency.
    """

    prefill_ms_per_token: float = 0.35
    decode_ms_per_token: float = 2.4
    parallel_efficiency: float = 0.62
    batch_overhead_ms: float = 2.5
    jitter_fraction: float = 0.03
    sleep_for_runtime: bool = True
    time_source: Callable[[], float] | None = None
    simulate_elapsed_when_not_sleeping: bool = True

    def execute_batch(
        self,
        batch: Sequence[GenerationRequest],
    ) -> tuple[list[GenerationResult], BatchExecutionMetrics]:
        if not batch:
            empty_metrics = BatchExecutionMetrics(
                batch_size=0,
                runtime_ms=0.0,
                prompt_tokens=0,
                generated_tokens=0,
            )
            return [], empty_metrics

        clock = self.time_source or time.monotonic
        start = clock()
        batch_size = len(batch)
        prompt_tokens = sum(item.prompt_tokens for item in batch)
        generated_tokens = sum(item.max_new_tokens for item in batch)

        speedup = 1.0 + (batch_size - 1) * self.parallel_efficiency
        runtime_ms = (
            (prompt_tokens * self.prefill_ms_per_token)
            + (generated_tokens * self.decode_ms_per_token)
        ) / speedup + self.batch_overhead_ms

        if self.jitter_fraction > 0:
            low = max(0.0, 1.0 - self.jitter_fraction)
            high = 1.0 + self.jitter_fraction
            runtime_ms *= random.uniform(low, high)

        if self.sleep_for_runtime:
            time.sleep(runtime_ms / 1000.0)
            end = clock()
        elif self.simulate_elapsed_when_not_sleeping:
            end = start + runtime_ms / 1000.0
        else:
            end = clock()
        total_latency_ms = (end - start) * 1000.0
        results: list[GenerationResult] = []
        for req in batch:
            queue_wait_ms = (start - req.submitted_at) * 1000.0
            e2e_ms = (end - req.submitted_at) * 1000.0
            text = f"[mock] request={req.request_id} tokens={req.max_new_tokens}"
            results.append(
                GenerationResult(
                    request_id=req.request_id,
                    workload=req.workload,
                    output_text=text,
                    queue_wait_ms=max(0.0, queue_wait_ms),
                    end_to_end_latency_ms=max(total_latency_ms, e2e_ms),
                )
            )

        metrics = BatchExecutionMetrics(
            batch_size=batch_size,
            runtime_ms=runtime_ms,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
        )
        return results, metrics


import llama_cpp
from typing import Dict, List, Optional

@dataclass
class LlamaCppIterationEngine:
    """
    Advanced engine implementing Continuous Batching (Iteration-level decoding).
    Uses llama.cpp low-level API for precise control.
    """
    model_path: str
    n_ctx: int = 2048
    n_threads: int = 8
    
    _model: Optional[llama_cpp.llama_model] = field(init=False, default=None)
    _ctx: Optional[llama_cpp.llama_context] = field(init=False, default=None)
    _batch: Optional[llama_cpp.llama_batch] = field(init=False, default=None)
    
    def __post_init__(self):
        params = llama_cpp.llama_model_default_params()
        self._model = llama_cpp.llama_load_model_from_file(self.model_path.encode(), params)
        
        ctx_params = llama_cpp.llama_context_default_params()
        ctx_params.n_ctx = self.n_ctx
        ctx_params.n_threads = self.n_threads
        ctx_params.n_seq_max = 16 # Support up to 16 concurrent sequences
        self._ctx = llama_cpp.llama_new_context_with_model(self._model, ctx_params)
        
        # Initialize a batch that can handle up to 512 tokens at once
        # n_tokens, n_embd, n_seq_max
        self._batch = llama_cpp.llama_batch_init(512, 0, 16)

    def step(self, active_requests: list[GenerationRequest]) -> list[GenerationResult]:
        """
        Processes ONE iteration (one token) for all active requests.
        Returns results for any requests that finished in this step.
        """
        if not active_requests:
            return []

        finished_results = []
        
        # Prepare the llama_batch
        self._batch.n_tokens = 0
        
        for i, req in enumerate(active_requests):
            if req.is_finished:
                continue
                
            # If this is a new request, we need to prefill
            if req.tokens_generated == 0:
                text_bytes = req.prompt.encode()
                n_max_tokens = len(text_bytes) + 2
                tokens = (llama_cpp.llama_token * n_max_tokens)()
                n_tokens = llama_cpp.llama_tokenize(
                    llama_cpp.llama_model_get_vocab(self._model),
                    text_bytes,
                    len(text_bytes),
                    tokens,
                    n_max_tokens,
                    True,
                    True
                )
                if n_tokens < 0:
                    # Retry with larger buffer if needed
                    n_tokens = abs(n_tokens)
                    tokens = (llama_cpp.llama_token * n_tokens)()
                    llama_tokenize(llama_cpp.llama_model_get_vocab(self._model), text_bytes, len(text_bytes), tokens, n_tokens, True, True)
                
                req.token_ids = [tokens[i] for i in range(n_tokens)]
                req.kv_cache_seq_id = i # Simple sequence mapping
                
                # Add prompt tokens to batch
                for j, tid in enumerate(req.token_ids):
                    pos = self._batch.n_tokens
                    self._batch.token[pos] = tid
                    self._batch.pos[pos] = j
                    self._batch.n_seq_id[pos] = 1
                    self._batch.seq_id[pos][0] = req.kv_cache_seq_id
                    self._batch.logits[pos] = (j == len(req.token_ids) - 1)
                    self._batch.n_tokens += 1
            else:
                # Add just the last generated token for decoding
                last_token = req.token_ids[-1]
                pos_in_seq = len(req.token_ids) - 1
                batch_pos = self._batch.n_tokens
                self._batch.token[batch_pos] = last_token
                self._batch.pos[batch_pos] = pos_in_seq
                self._batch.n_seq_id[batch_pos] = 1
                self._batch.seq_id[batch_pos][0] = req.kv_cache_seq_id
                self._batch.logits[batch_pos] = True
                self._batch.n_tokens += 1

        # Decode the batch
        if self._batch.n_tokens > 0:
            llama_cpp.llama_decode(self._ctx, self._batch)
            
            # Get all logits for the batch
            all_logits = llama_cpp.llama_get_logits(self._ctx)
            n_vocab = llama_cpp.llama_n_vocab(self._model)
            
            import numpy as np
            # Total number of tokens in batch that requested logits
            n_logits = sum(1 for i in range(self._batch.n_tokens) if self._batch.logits[i])
            logits_arr = np.ctypeslib.as_array(all_logits, shape=(n_logits, n_vocab))
            
            logit_idx = 0
            for i, req in enumerate(active_requests):
                if req.is_finished: continue
                
                # The logits for this request are at logit_idx
                req_logits = logits_arr[logit_idx]
                logit_idx += 1
                
                new_token_id = int(np.argmax(req_logits))
                
                req.token_ids.append(new_token_id)
                req.tokens_generated += 1
                
                # Check for finish condition
                if new_token_id == llama_cpp.llama_token_eos(self._model) or req.tokens_generated >= req.max_new_tokens:
                    req.is_finished = True
                    text = llama_cpp.llama_token_to_piece(self._model, new_token_id).decode() # Simplification
                    req.generated_text += text
                    
                    # Create result
                    now = time.monotonic()
                    finished_results.append(GenerationResult(
                        request_id=req.request_id,
                        workload=req.workload,
                        output_text=req.generated_text,
                        queue_wait_ms=(now - req.submitted_at) * 1000.0, # Approximate
                        end_to_end_latency_ms=(now - req.submitted_at) * 1000.0,
                        total_tokens=req.tokens_generated
                    ))
                else:
                    text = llama_cpp.llama_token_to_piece(self._model, new_token_id).decode()
                    req.generated_text += text

        return finished_results

    def __del__(self):
        if self._batch: llama_cpp.llama_batch_free(self._batch)
        if self._ctx: llama_cpp.llama_free(self._ctx)
        if self._model: llama_cpp.llama_free_model(self._model)
@dataclass(slots=True)
class MlxEngineAdapter:
    """
    Adapter for integrating the scheduler with MLX-LM execution.
    """

    model: any
    tokenizer: any

    def execute_batch(
        self,
        batch: Sequence[GenerationRequest],
    ) -> tuple[list[GenerationResult], BatchExecutionMetrics]:
        if not batch:
            empty = BatchExecutionMetrics(0, 0.0, 0, 0)
            return [], empty

        import mlx_lm
        
        start = time.monotonic()
        
        # MLX-LM doesn't have a high-level "batch_generate" that perfectly matches 
        # our GenerationRequest structure easily, so we loop for now, 
        # or use internal batched decode if possible.
        # For simplicity and latency capture, we process them.
        
        outputs = []
        total_generated_tokens = 0
        total_prompt_tokens = 0
        
        for req in batch:
            # We use mlx_lm.generate
            # In a real high-throughput scenario, we'd use batched inputs.
            output = mlx_lm.generate(
                self.model,
                self.tokenizer,
                prompt=req.prompt,
                max_tokens=req.max_new_tokens,
                verbose=False
            )
            outputs.append(output)
            # Rough estimate of tokens (word count for now, tokenizer length better)
            total_generated_tokens += len(output.split())
            total_prompt_tokens += req.prompt_tokens

        end = time.monotonic()
        runtime_ms = (end - start) * 1000.0

        results: list[GenerationResult] = []
        for req, text in zip(batch, outputs):
            queue_wait_ms = (start - req.submitted_at) * 1000.0
            e2e_ms = (end - req.submitted_at) * 1000.0
            results.append(
                GenerationResult(
                    request_id=req.request_id,
                    workload=req.workload,
                    output_text=text,
                    queue_wait_ms=max(0.0, queue_wait_ms),
                    end_to_end_latency_ms=max(0.0, e2e_ms),
                )
            )

        metrics = BatchExecutionMetrics(
            batch_size=len(batch),
            runtime_ms=max(0.0, runtime_ms),
            prompt_tokens=total_prompt_tokens,
            generated_tokens=total_generated_tokens,
        )
        return results, metrics
