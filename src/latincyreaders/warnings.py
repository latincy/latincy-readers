"""Warning categories emitted by latincy-readers.

These are ``UserWarning`` subclasses so they surface by default, and can be
filtered or escalated to errors individually via :mod:`warnings`, e.g.::

    import warnings
    from latincyreaders.warnings import AnnotationModelMismatchWarning
    warnings.simplefilter("error", AnnotationModelMismatchWarning)
"""

from __future__ import annotations


class AnnotationModelMismatchWarning(UserWarning):
    """Cached annotations were produced by a different model than the active one.

    Emitted when a canonical ``.conlluc`` file records a ``model_name`` /
    ``model_version`` that differs from the model currently in use. The
    annotations may be stale (e.g. sentence boundaries produced by an older
    model), so downstream results could differ from a fresh run. The canonical
    store is deliberately *not* auto-invalidated — it may be intentionally
    pinned or community-corrected — so the caller decides whether to rebuild.
    """
