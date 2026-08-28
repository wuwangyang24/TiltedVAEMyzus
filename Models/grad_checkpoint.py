"""Gradient-checkpointing helper.

timm calls ``torch.utils.checkpoint.checkpoint`` without ``use_reentrant``,
which emits a UserWarning (and will raise in future PyTorch versions). This
helper enables checkpointing on a timm backbone while forcing the recommended
non-reentrant variant.
"""
import functools
import sys

import torch.utils.checkpoint as _cp

_PATCHED = False


def _patch_checkpoint_default() -> None:
    """Make ``use_reentrant=False`` the default for timm's checkpoint calls."""
    global _PATCHED
    if _PATCHED:
        return

    _orig_checkpoint = _cp.checkpoint

    @functools.wraps(_orig_checkpoint)
    def _checkpoint(*args, use_reentrant=False, **kwargs):
        return _orig_checkpoint(*args, use_reentrant=use_reentrant, **kwargs)

    # Patch the canonical location plus any module that imported the function
    # by reference (e.g. ``from torch.utils.checkpoint import checkpoint`` in
    # timm.models._manipulate), otherwise those bindings keep the original.
    _cp.checkpoint = _checkpoint
    for module in list(sys.modules.values()):
        if module is None:
            continue
        if getattr(module, "checkpoint", None) is _orig_checkpoint:
            module.checkpoint = _checkpoint
    _PATCHED = True


def enable_grad_checkpointing(backbone) -> None:
    """Enable activation checkpointing on a timm ``backbone`` (non-reentrant)."""
    _patch_checkpoint_default()
    backbone.set_grad_checkpointing(enable=True)
