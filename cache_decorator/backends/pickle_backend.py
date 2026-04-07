"""Backend for compressing and pickling objects."""

from typing import Dict
from pickle import dump as pickle_dump, load as pickle_load
from .backend_template import BackendTemplate


class PickleBackend(BackendTemplate):
    """Backend for compressing and pickling objects."""

    SUPPORTED_EXTENSIONS = [".pkl"]

    @staticmethod
    def support_path(path: str) -> bool:
        return any(
            path.endswith(extension) for extension in PickleBackend.SUPPORTED_EXTENSIONS
        )

    @staticmethod
    def can_serialize(obj_to_serialize: object, path: str) -> bool:
        return PickleBackend.support_path(path)

    @staticmethod
    def can_deserialize(metadata: Dict, path: str) -> bool:
        return PickleBackend.support_path(path)

    def dump(self, obj_to_serialize: object, path: str) -> Dict:  # type: ignore[reportReturnType]
        with open(path, "wb") as f:
            pickle_dump(obj_to_serialize, f, **self._dump_kwargs)

    def load(self, metadata: Dict, path: str) -> object:
        with open(path, "rb") as f:
            return pickle_load(f, **self._load_kwargs)
