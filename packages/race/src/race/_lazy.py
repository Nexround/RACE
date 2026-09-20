"""Shared lazy-import helper for the race package.

Each sub-package ``__init__.py`` can use :func:`load_lazy_attr` to implement
``__getattr__`` without duplicating the import machinery.

Design goals
------------
* A single, well-tested implementation instead of per-package copies.
* Attribute value is cached in the calling module's ``globals()`` so that
  the second access costs only a dict lookup, never another ``__getattr__``
  call.
* Missing attributes raise :exc:`AttributeError` with a clean message –
  no silent ``None`` returns.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


def load_lazy_attr(
    module_globals: dict[str, Any],
    package: str,
    import_map: dict[str, tuple[str, str]],
    name: str,
) -> Any:
    """Resolve *name* via *import_map* and cache it in *module_globals*.

    Parameters
    ----------
    module_globals:
        The ``globals()`` dict of the calling ``__init__.py``.  The resolved
        value is written back here so subsequent accesses skip ``__getattr__``
        entirely.
    package:
        The fully-qualified package name of the caller (i.e. ``__name__``).
        Used both for the error message and as the *package* argument to
        :func:`importlib.import_module` for relative paths.
    import_map:
        Mapping from exported name → ``(module_path, attribute_name)``.
        *module_path* may be absolute (``"race.core.analyzer"``) or relative
        (``".core.analyzer"``); relative paths are resolved against *package*.
    name:
        The attribute being looked up.

    Returns
    -------
    Any
        The resolved attribute value.

    Raises
    ------
    AttributeError
        If *name* is not found in *import_map*.
    """
    try:
        module_name, attr_name = import_map[name]
    except KeyError:
        raise AttributeError(f"module {package!r} has no attribute {name!r}") from None

    module = import_module(module_name, package)
    value = getattr(module, attr_name)
    # Cache so the next access is a plain dict lookup.
    module_globals[name] = value
    return value
