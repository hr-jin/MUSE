import os, threading, time
import numpy as np
from openai import OpenAI


class RawEmbedder:
    def __init__(self, url, model, batch_size=8, timeout=45, api_key=None):
        self.url = url
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.api_key = api_key or os.environ.get("MUSE_EMBEDDING_API_KEY", "local")
        self.cache = {}
        self.lock = threading.Lock()

    def embed(self, texts):
        missing = list(dict.fromkeys(t for t in texts if t not in self.cache))
        usage = []
        start = time.time()
        if missing:
            with self.lock:
                missing = [t for t in missing if t not in self.cache]
                with OpenAI(
                    api_key=self.api_key,
                    base_url=self.url,
                    timeout=self.timeout,
                    max_retries=0,
                ) as client:
                    for offset in range(0, len(missing), self.batch_size):
                        batch = missing[offset : offset + self.batch_size]
                        r = client.embeddings.create(model=self.model, input=batch)
                        data = sorted(r.data, key=lambda x: x.index)
                        if [x.index for x in data] != list(range(len(batch))):
                            raise ValueError("Embedding response index mismatch")
                        vs = np.asarray([x.embedding for x in data], dtype=np.float32)
                        norms = np.linalg.norm(vs, axis=1, keepdims=True)
                        if not np.isfinite(vs).all() or not (norms > 0).all():
                            raise ValueError("Invalid embedding vectors")
                        vs /= norms
                        for text, v in zip(batch, vs):
                            self.cache[text] = v
                        if r.usage:
                            usage.append(r.usage.model_dump())
        return np.asarray([self.cache[t] for t in texts], dtype=np.float32), {
            "new_blocks": len(missing),
            "cache_hit_blocks": len(texts) - len(missing),
            "usage": usage,
            "elapsed_seconds": time.time() - start,
            "instruction_prefix_used": False,
        }
