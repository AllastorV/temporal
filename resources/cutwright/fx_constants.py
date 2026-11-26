"""Numbers the frame kernels and Cutwright's own process both need. Imports nothing, so the
render's main process reads them (an effect's warm-up, ``FxDef.history``) without importing a
kernel file."""

CARRY = 12  # frames between the resets of a carried texture layer (= the warm-up it needs)
JITTER = 0.0015  # boil: registration jitter per frame, short-side fraction (~1.6 px at 1080p)
# A job's error text (lower case) that the CPU can still do the job after; not every "cuda" text:
# a device mismatch is a bug, and would hide behind a slower render.
GPU_LOST = ("out of memory", "cufft_alloc_failed", "cudnn_status_not_enough_memory",
            "cuda error: unknown", "illegal memory")  # fmt: skip
