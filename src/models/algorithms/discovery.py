"""Discover concrete classifier plugins only inside the trusted algorithm package."""
from importlib import import_module
from inspect import isabstract, isclass
from pkgutil import iter_modules
import re

from .base import BaseAlgorithm


def discover_plugins(package):
    """Find one concrete plugin per local module with deterministic registration order."""
    found = []
    for item in sorted(iter_modules(package.__path__), key=lambda entry: entry.name):
        # Do not import external entry points or derive imports from experiment YAML.
        if item.ispkg or item.name.startswith('_') or item.name in {'base', 'registry', 'sklearn_base', 'discovery'}:
            continue
        if not re.fullmatch(r'[a-z][a-z0-9_]*', item.name):
            raise ValueError(f"Invalid algorithm module name: {item.name}")
        module = import_module(f'{package.__name__}.{item.name}')
        candidates = [value for value in vars(module).values() if isclass(value)
                      and value.__module__ == module.__name__ and issubclass(value, BaseAlgorithm)
                      and not isabstract(value)]
        if not candidates:
            continue  # Shared utility modules are not cartridges.
        if len(candidates) != 1:
            raise ValueError(f"Expected one concrete algorithm in {module.__name__}")
        plugin = candidates[0]()
        if plugin.name != item.name:
            raise ValueError(f"Plugin identifier {plugin.name!r} must match {item.name!r}")
        order = getattr(plugin, 'registration_order', 1000)
        if type(order) is not int:
            raise ValueError(f"Invalid registration_order for {plugin.name}")
        found.append((order, plugin.name, plugin))
    return [plugin for _, _, plugin in sorted(found, key=lambda entry: entry[:2])]
