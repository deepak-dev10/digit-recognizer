"""Look up a model backend by name and load it once.

Usage::

    from digitrec.config import Config
    from digitrec.models.registry import get_predictor

    predictor = get_predictor(Config(backend="pretrained"))
    predictor.predict(crops)

How backends are registered
---------------------------
Each backend module provides a **factory**: a function ``load(config: Config) -> Predictor``
that builds the model, loads its weights and moves it to ``config.device``.

Built-in backends are listed in ``_BACKENDS`` as ``"module:function"`` strings and are
**imported only when first requested**. That keeps start-up fast and means a backend's
heavy dependencies (e.g. ``transformers`` for the pretrained model) load only if you use it.

Caching
-------
Loading a model (downloading weights, moving them to the GPU) is slow, so
:func:`get_predictor` caches one predictor per distinct :class:`~digitrec.config.Config`.
Calling it again with an equal config returns the *same* object.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from functools import lru_cache

from digitrec.config import Config
from digitrec.models.base import Predictor

#: A function that builds a ready-to-use predictor from a config.
Factory = Callable[[Config], Predictor]

# Built-in backends: name -> "module:function" (imported lazily) or a factory callable.
_BACKENDS: dict[str, str | Factory] = {
    "pretrained": "digitrec.models.pretrained:load",  # Phase 1: Hugging Face MNIST model
    "scratch_cnn": "digitrec.models.cnn:load",        # Phase 2: our own CNN + checkpoint
}


def available_backends() -> list[str]:
    """Names accepted by :func:`get_predictor` (e.g. for CLI choices or a UI selector)."""
    return sorted(_BACKENDS)


def register_backend(name: str, factory: str | Factory) -> None:
    """Add or replace a backend.

    Args:
        name: Name to use in ``Config.backend``.
        factory: A factory callable, or a ``"module:function"`` string imported on first use.

    Clears the predictor cache, so a replaced backend is never served from a stale entry.
    """
    _BACKENDS[name] = factory
    clear_cache()


def _resolve_factory(name: str) -> Factory:
    """Return the factory for ``name``, importing its module if needed."""
    try:
        target = _BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"Unknown backend {name!r}; available: {', '.join(available_backends())}"
        ) from None

    if callable(target):
        return target

    module_name, _, func_name = target.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as e:
        # Only rewrite the error when the backend module itself is missing (e.g. Phase 2
        # code not written yet). A missing *dependency* inside it is re-raised unchanged.
        if e.name != module_name:
            raise
        raise NotImplementedError(
            f"Backend {name!r} is registered but its module {module_name!r} does not exist yet"
        ) from e
    return getattr(module, func_name)


@lru_cache(maxsize=None)
def get_predictor(config: Config | None = None) -> Predictor:
    """Return the predictor for ``config.backend``, loading it on the first call.

    Args:
        config: Settings to use; ``None`` means ``Config()`` (default backend, auto device).

    Returns:
        A loaded :class:`~digitrec.models.base.Predictor`, cached per config.

    Raises:
        ValueError: ``config.backend`` is not a registered name.
        NotImplementedError: The backend is registered but its module isn't written yet.
    """
    config = config or Config()
    return _resolve_factory(config.backend)(config)


def clear_cache() -> None:
    """Forget all loaded predictors (frees model memory; mainly for tests)."""
    get_predictor.cache_clear()
