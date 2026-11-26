"""Explicit registration and lookup of algorithm plugins."""

from .base import BaseAlgorithm


class AlgorithmRegistry:
    """Map YAML algorithm identifiers to reusable plugin instances."""

    def __init__(self) -> None:
        self._plugins: dict[str, BaseAlgorithm] = {}

    def register(self, plugin: BaseAlgorithm) -> None:
        """Register a plugin and reject invalid or duplicate identifiers."""
        if not isinstance(plugin, BaseAlgorithm):
            raise TypeError("Plugin must implement BaseAlgorithm.")

        name = getattr(plugin, "name", None)
        tasks = getattr(plugin, "supported_tasks", None)

        if not isinstance(name, str) or not name or name != name.strip():
            raise ValueError("Plugin must declare a valid name.")

        if not isinstance(tasks, tuple) or not tasks or not all(
            isinstance(task, str) and task.strip() for task in tasks
        ):
            raise ValueError("Plugin must declare supported tasks.")

        if name in self._plugins:
            raise ValueError(f"Algorithm already registered: {name}")

        self._plugins[name] = plugin

    def get(
        self, name: str, *, task_type: str | None = None
    ) -> BaseAlgorithm:
        """Retrieve a plugin, optionally checking task compatibility."""
        if name not in self._plugins:
            raise KeyError(f"Unknown algorithm: {name}")

        plugin = self._plugins[name]

        if task_type is not None and task_type not in plugin.supported_tasks:
            raise ValueError(f"Algorithm {name} does not support {task_type}")

        return plugin

    def names(self) -> tuple[str, ...]:
        """Return registered identifiers in registration order."""
        return tuple(self._plugins)


# Production plugins will be registered explicitly during migration.
ALGORITHM_REGISTRY = AlgorithmRegistry()