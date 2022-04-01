import qtensor_ai
import contextlib

@contextlib.contextmanager
def forward_only():
    prev = qtensor_ai.forward_only_reuse_memory
    qtensor_ai.forward_only_reuse_memory = True
    yield
    qtensor_ai.forward_only_reuse_memory = prev