from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import llama_cpp
    import numpy as np
    _HAS_LLAMA = True
except ImportError:
    _HAS_LLAMA = False

from .models import GenerationRequest, GenerationResult

logger = logging.getLogger(__name__)


@dataclass
class LlamaCppIterationEngine:
    """
    Production-grade engine implementing Continuous Batching (Iteration-level decoding).
    Uses llama.cpp low-level API for precise control over token-by-token execution.
    """
    model_path: str
    n_ctx: int = 2048
    n_threads: int = 8
    n_batch: int = 512
    max_sequences: int = 16

    _model: Optional[llama_cpp.llama_model] = field(init=False, default=None)
    _ctx: Optional[llama_cpp.llama_context] = field(init=False, default=None)
    _batch: Optional[llama_cpp.llama_batch] = field(init=False, default=None)
    _lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    
    # Track available slots in the KV cache sequence space
    _free_slots: List[int] = field(init=False)

    def __post_init__(self):
        if not _HAS_LLAMA:
            raise ImportError(
                "llama-cpp-python and numpy are required for LlamaCppIterationEngine. "
                "Install them with: pip install llama-cpp-python numpy"
            )

        params = llama_cpp.llama_model_default_params()
        self._model = llama_cpp.llama_load_model_from_file(self.model_path.encode(), params)
        if not self._model:
            raise RuntimeError(f"Failed to load model from {self.model_path}")

        ctx_params = llama_cpp.llama_context_default_params()
        ctx_params.n_ctx = self.n_ctx
        ctx_params.n_threads = self.n_threads
        ctx_params.n_batch = self.n_batch
        
        # Enable multiple sequences in the KV cache
        self._ctx = llama_cpp.llama_new_context_with_model(self._model, ctx_params)
        if not self._ctx:
            raise RuntimeError("Failed to create llama_context")

        # Initialize a batch structure
        # n_tokens_max, n_embd, n_seq_max
        self._batch = llama_cpp.llama_batch_init(self.n_batch, 0, self.max_sequences)
        
        # Initialize free slots for sequence IDs (0 to max_sequences - 1)
        self._free_slots = list(range(self.max_sequences))

    def step(self, active_requests: List[GenerationRequest]) -> List[GenerationResult]:
        """
        Processes ONE iteration (token decode) for all active requests.
        Injects new requests (prefill) and advances existing ones (decode).
        Returns results for any requests that finished in this step.
        """
        if not active_requests:
            return []

        finished_results = []
        
        with self._lock:
            # 1. Reset batch
            self._batch.n_tokens = 0
            
            requests_to_process = []
            
            for req in active_requests:
                if req.is_finished:
                    continue
                
                # Assign a KV cache sequence ID if not already assigned
                if req.kv_cache_seq_id == -1:
                    if not self._free_slots:
                        logger.warning(f"No free KV cache slots for request {req.request_id}. Skipping.")
                        continue
                    req.kv_cache_seq_id = self._free_slots.pop(0)

                # 2. Add tokens to batch
                if req.tokens_generated == 0:
                    # PREFILL path
                    text_bytes = req.prompt.encode("utf-8")
                    # Tokenize the prompt
                    n_tokens_max = len(req.prompt) + 4
                    tokens = (llama_cpp.llama_token * n_tokens_max)()
                    n_tokens = llama_cpp.llama_tokenize(
                        llama_cpp.llama_model_get_vocab(self._model),
                        text_bytes,
                        len(text_bytes),
                        tokens,
                        n_tokens_max,
                        True,  # Add BOS
                        True   # Special tokens
                    )
                    
                    if n_tokens < 0:
                        # Re-allocate and try again if buffer was too small
                        n_tokens = abs(n_tokens)
                        tokens = (llama_cpp.llama_token * n_tokens)()
                        llama_cpp.llama_tokenize(
                            llama_cpp.llama_model_get_vocab(self._model),
                            text_bytes,
                            len(text_bytes),
                            tokens,
                            n_tokens,
                            True,
                            True
                        )
                    
                    req.token_ids = [tokens[i] for i in range(n_tokens)]
                    
                    # Add all prompt tokens to batch for prefill
                    for i, tid in enumerate(req.token_ids):
                        pos = self._batch.n_tokens
                        self._batch.token[pos] = tid
                        self._batch.pos[pos] = i
                        self._batch.n_seq_id[pos] = 1
                        self._batch.seq_id[pos][0] = req.kv_cache_seq_id
                        # Only request logits for the very last token of the prompt
                        self._batch.logits[pos] = (i == len(req.token_ids) - 1)
                        self._batch.n_tokens += 1
                        
                        if self._batch.n_tokens >= self.n_batch:
                            # We hit the batch limit, stop adding more tokens
                            break
                else:
                    # DECODE path (single token)
                    last_token = req.token_ids[-1]
                    pos_in_seq = len(req.token_ids) - 1
                    batch_pos = self._batch.n_tokens
                    
                    self._batch.token[batch_pos] = last_token
                    self._batch.pos[batch_pos] = pos_in_seq
                    self._batch.n_seq_id[batch_pos] = 1
                    self._batch.seq_id[batch_pos][0] = req.kv_cache_seq_id
                    self._batch.logits[batch_pos] = True
                    self._batch.n_tokens += 1
                
                requests_to_process.append(req)
                if self._batch.n_tokens >= self.n_batch:
                    break

            # 3. Execute the batch (decode/prefill)
            if self._batch.n_tokens > 0:
                ret = llama_cpp.llama_decode(self._ctx, self._batch)
                if ret != 0:
                    logger.error(f"llama_decode failed with error code {ret}")
                    return []
                
                # 4. Extract logits and sample new tokens
                all_logits = llama_cpp.llama_get_logits(self._ctx)
                try:
                    # Use llama_vocab_n_tokens for compatibility
                    n_vocab = llama_cpp.llama_vocab_n_tokens(llama_cpp.llama_model_get_vocab(self._model))
                except AttributeError:
                    # Fallback for older llama-cpp-python versions
                    n_vocab = llama_cpp.llama_n_vocab(self._model)
                
                # Identify which requests in the batch produced logits
                # (Only those where self._batch.logits[i] was True)
                logit_indices = [i for i in range(self._batch.n_tokens) if self._batch.logits[i]]
                
                for idx, req in enumerate(requests_to_process):
                    # Find which logit set belongs to this request
                    # Since we only request 1 logit per request per step, 
                    # the order in all_logits corresponds to the order in logit_indices
                    if idx >= len(logit_indices):
                        continue
                        
                    l_idx = logit_indices[idx]
                    # Map to the numpy array of logits
                    req_logits = np.ctypeslib.as_array(all_logits + (idx * n_vocab), shape=(n_vocab,))
                    
                    # Greedy sampling for production reliability (can be extended with top-p/k)
                    new_token_id = int(np.argmax(req_logits))
                    
                    # Update request state
                    if req.tokens_generated == 0:
                        # First token after prefill doesn't add to token_ids (it's already there)
                        # We just need to check it and start generating
                        pass
                    
                    req.token_ids.append(new_token_id)
                    req.tokens_generated += 1
                    
                    # Detokenize and append to output string
                    piece = self._detokenize_token(new_token_id)
                    req.generated_text += piece
                    
                    # Check for completion
                    is_eos = (new_token_id == llama_cpp.llama_token_eos(self._model))
                    is_limit = (req.tokens_generated >= req.max_new_tokens)
                    
                    if is_eos or is_limit:
                        req.is_finished = True
                        
                        # Cleanup KV cache for this sequence
                        llama_cpp.llama_kv_cache_seq_rm(self._ctx, req.kv_cache_seq_id, -1, -1)
                        self._free_slots.append(req.kv_cache_seq_id)
                        
                        now = time.monotonic()
                        finished_results.append(GenerationResult(
                            request_id=req.request_id,
                            workload=req.workload,
                            output_text=req.generated_text.strip(),
                            queue_wait_ms=(now - req.submitted_at) * 1000.0,
                            end_to_end_latency_ms=(now - req.submitted_at) * 1000.0,
                            total_tokens=req.tokens_generated
                        ))

        return finished_results

    def _detokenize_token(self, token_id: int) -> str:
        """Safe detokenization of a single token."""
        try:
            # Buffer for the token piece
            buffer = (llama_cpp.c_char * 128)()
            n = llama_cpp.llama_token_to_piece(self._model, token_id, buffer, len(buffer), 0, True)
            if n > 0:
                return bytes(buffer[:n]).decode("utf-8", errors="replace")
        except Exception as e:
            logger.error(f"Detokenization failed for token {token_id}: {e}")
        return ""

    def __del__(self):
        # Cleanup C structures
        if hasattr(self, '_batch') and self._batch:
            llama_cpp.llama_batch_free(self._batch)
        if hasattr(self, '_ctx') and self._ctx:
            llama_cpp.llama_free(self._ctx)
        if hasattr(self, '_model') and self._model:
            llama_cpp.llama_free_model(self._model)
