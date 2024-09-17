"""Backend for json files."""

from typing import Dict
import compress_json
from .backend_template import BackendTemplate


class JsonBackend(BackendTemplate):
    """Backend for json files."""

    SUPPORTED_EXTENSIONS = [".json"]

    @staticmethod
    def support_path(path: str) -> bool:
        return any(
            path.endswith(extension) for extension in JsonBackend.SUPPORTED_EXTENSIONS
        )

    @staticmethod
    def can_deserialize(metadata: Dict, path: str) -> bool:
        """Must return if the current backend can handle the type of data."""
        return JsonBackend.support_path(path)

    @staticmethod
    def can_serialize(obj_to_serialize: object, path: str) -> bool:
        """Returns if we can serialize the given type as the given extension"""
        return JsonBackend.support_path(path)

    def dump(self, obj_to_serialize: object, path: str) -> Dict:
        """Serialize and save the object at the given path.
        If this backend needs extra informations to de-serialize data, it can
        return them as a dictionary which will be serialized as a json."""
        compress_json.dump(obj_to_serialize, path, json_kwargs=self._dump_kwargs)

    def load(self, metadata: Dict, path: str) -> object:
        """Load the method at the given path. If the medod need extra
        informations it can read them form the metadata dictionary which is
        the return value of the dump method."""
        return compress_json.load(path, json_kwargs=self._load_kwargs)
