"""Backend for compress_json."""

from typing import Dict
import compress_json
from .backend_template import BackendTemplate


class CompressJsonBackend(BackendTemplate):
    """Backend for compress_json."""

    SUPPORTED_EXTENSIONS = [
        ".json",
        ".json.gz",
        ".json.bz",
        ".json.lzma",
    ]

    @staticmethod
    def support_path(path: str) -> bool:
        return any(
            path.endswith(extension)
            for extension in CompressJsonBackend.SUPPORTED_EXTENSIONS
        )

    @staticmethod
    def can_deserialize(metadata: Dict, path: str) -> bool:
        return CompressJsonBackend.support_path(path)

    @staticmethod
    def can_serialize(obj_to_serialize: object, path: str) -> bool:
        return CompressJsonBackend.can_deserialize({}, path)

    def dump(self, obj_to_serialize: object, path: str) -> Dict:  # type: ignore[reportReturnType]
        compress_json.dump(obj_to_serialize, path, **self._dump_kwargs)

    def load(self, metadata: Dict, path: str) -> object:
        return compress_json.load(path, **self._load_kwargs)
