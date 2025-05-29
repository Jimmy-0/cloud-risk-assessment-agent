import os

class PromptManager:
    """Cache and return text prompts from the prompts directory."""

    def __init__(self, base_path: str = "./src/prompts"):
        self.base_path = base_path
        self._cache: dict[str, str] = {}

    def load(self, name: str) -> str:
        """Return the content of ``{name}_prompt.txt`` from ``base_path``."""
        if name not in self._cache:
            path = os.path.join(self.base_path, f"{name}_prompt.txt")
            with open(path, "r", encoding="utf-8") as f:
                self._cache[name] = f.read()
        return self._cache[name]
