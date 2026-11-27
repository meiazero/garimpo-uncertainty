"""A config names a model, a loss or any other swappable part; its registry builds it."""

import json
from collections.abc import Callable
from typing import Any


class Registry:
    """Entries by name, filled with `@registry.register(name)` where each entry is defined."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.entries: dict[str, Callable] = {}

    def register[T: Callable](self, name: str) -> Callable[[T], T]:
        def add(entry: T) -> T:
            if name in self.entries:
                raise ValueError(f"{self.kind} {name!r} is registered twice")
            self.entries[name] = entry
            return entry

        return add

    def __getitem__(self, name: str) -> Callable:
        if name not in self.entries:
            raise ValueError(f"unknown {self.kind} {name!r}; known: {', '.join(self.names())}")
        return self.entries[name]

    def names(self) -> list[str]:
        return sorted(self.entries)

    def entry(self, spec: str | dict[str, Any]) -> Callable:
        """Return what `spec` names, without building it."""
        return self[spec if isinstance(spec, str) else spec["name"]]

    def build(self, spec: str | dict[str, Any], **arguments: Any) -> Any:
        """Build what `spec` names: a bare name, or {"name": ..., **options} from a config."""
        options = {"name": spec} if isinstance(spec, str) else dict(spec)
        return self[options.pop("name")](**options, **arguments)


def parse_spec(text: str) -> str | dict[str, Any]:
    """Read a command-line spec: a bare name, or a JSON object with a name and options."""
    if not text.startswith("{"):
        return text
    spec = json.loads(text)
    if not isinstance(spec, dict) or not isinstance(spec.get("name"), str):
        raise ValueError(f"a spec object needs a name: {text}")  # noqa: TRY004 - argparse reads ValueError
    return spec
