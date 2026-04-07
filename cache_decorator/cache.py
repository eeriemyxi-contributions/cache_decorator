"""This module contains the Cache class that is used to cache the results of a function."""

import os
import sys
import json
import pickle
import typing as t

import inspect
import logging
from time import time
from functools import wraps
from datetime import datetime
import humanize

# Dictionary are not hashable and the python hash is not consistent
# between runs so we have to use an external dictionary hashing package
# else we will not be able to load the saved caches.
from dict_hash import sha256, Hashable

from .utils import (
    get_params,
    parse_time,
    random_string,
    get_format_groups,
    get_next_format_group,
    get_function_name,
)
from .backends import Backend

P = t.ParamSpec("P")
R = t.TypeVar("R")

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARN,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "crit": logging.CRITICAL,
}


def cache(function: t.Callable[P, R]) -> t.Callable[P, R]:
    """Cache with default parameters"""
    return Cache()(function)


class Cache:
    """Cache the results of a function (or method)."""
    
    log_level: str
    log_format: str
    args_to_ignore: list[str]
    use_source_code: bool
    validity_duration: int | float | None
    use_approximated_hash: bool
    cache_path: str | tuple[t.Any, ...] | list[t.Any] | dict[str, t.Any]
    is_backup_enabled: bool
    backup_path: str | None
    cache_dir: str
    logger: logging.Logger | None
    function_info: dict[str, t.Any] | None
    decorated_function: t.Callable[..., t.Any] | None
    load_kwargs: dict[str, t.Any]
    dump_kwargs: dict[str, t.Any]
    enable_cache_arg_name: str | None
    capture_enable_cache_arg_name: bool
    optional_path_keys: list[str]

    def __init__(
        self,
        cache_path: str | tuple[t.Any, ...] | list[t.Any] | dict[str, t.Any] = "{cache_dir}/{function_name}/{_hash}.pkl",
        args_to_ignore: tuple[str, ...] = (),
        cache_dir: str | None = None,
        validity_duration: int | str | None = -1,
        use_source_code: bool = True,
        use_approximated_hash: bool = False,
        log_level: str = "critical",
        log_format: str = "%(asctime)-15s [%(levelname)s]: %(message)s",
        backup_path: str | None = None,
        backup: bool = True,
        optional_path_keys: list[str] | None = None,
        dump_kwargs: dict[str, t.Any] | None = None,
        load_kwargs: dict[str, t.Any] | None = None,
        enable_cache_arg_name: str | None = None,
        capture_enable_cache_arg_name: bool = True,
    ):
        """
        Cache the results of a function (or method).

        Example:
        ```
        from cache_decorator import Cache
        @Cache()
        def test(x):
            return 2 * x
        ```

        Arguments
        ---------
        cache_path: Union[str, Tuple[str], List[str], Dict[str, str]] = "{cache_dir}/{function_name}/{_hash}.pkl",
            Where to save the caches.
            It's a string format and the available variables are
            `cache_dir` the directory specified in the other argument.
            `function_name` the name of the cached function.
            `args_to_ignore` which arguments can be ignored form the input.
            `args` the name of the arguments (both positional and keyword).
            `defaults` the default values of the positional arguments.
            `kwonlydefaults` the default values of the kwarguments.
            `source` if `use_source_code` is setted to true, it's the string
                with the source code of the function to cache.
            `_hash` it's the hash of the parameters (excluded the ignored ones),
                this is computed only if it's present in `cache_path` so
                it's possible to cache functions which take non-hashable arguments.
            Moreover, you can use any argument passed to the function.
            Example:
            ```
            from cache_decorator import Cache
            @Cache("{cache_dir}/{x}/{y}.pkl)
            def test(x, y):
                return x * y
            ```
            The extension used in the format string determines the serialization method.
            The available ones are `.json .json.gz .json.bz .json.lzma .pkl .pkl.gz .pkl.bz
            .pkl.lzma .pkl.zip .npy .npz .csv .csv.gz .csv.bz2 .csv.zip .csv.xz .xlsx`
            This can also be used to make multiple arguments use the same cache:
            Example:
            ```
            from cache_decorator import Cache
            @Cache("{cache_dir}/{x}.pkl)
            def test(x, y):
                return x * y
            ```
            In this case the cache will be used watching only the `x` variable and
            the `y` is ignored. so `test(1, 2)` and `test(1, 10000)` will use the same
            cache (even if that's not right!). This can be used to save human readable
            partial results, in any other cases you should use the `_hash`.
        args_to_ignore: Tuple[str] = (),
            Which arguments to ignore when computing the hash.
        cache_dir: str = None,
            The folder where to save the caches. If not specified it read the value of
            the enviornment variable `CACHE_DIR`. If even this is empty it defaults to
            "./cache". This value is substituted in the `cache_path` argument if present.
        validity_duration: Union[int, str] = None,
            If not None, the cache will be recomputed after the specified ammount of time.
            This is done by saving a json with the same name of the cache plus `_time.json` which contains the
            computation epoch.
            If `validity_duration` is specified and a cache does not have it's json file, it's considered invalid.
            The given time must be an integer in seconds or a string in the format (\\d+[smhdw]) to specify
            a given ammount of s(econds), m(inutes), h(ours), d(ays), w(eeks).
        use_source_code: bool = True,
            If in the computing of the hash the must also use the sourcecode of the cached function.
        use_approximated_hash: bool = False,
            `dict_hash` exposes an `use_approximation` arg that allows for faster
            hashes at the cost of less precision because it only hashes part
            of the big parameters like big dataframes of numpy arrays.
        log_level: str = "critical",
            Set the logger level to the wanted level. The usable levels are:
            ["debug", "info", "warning", "error", "critical"]
            Alternatively a reference to the logger can be obtained with
            `logging.getLogger("cache." + function.__name__)`
            so it possible to fully customize it, like set the level and add filehandlers.
            Example:
            ```
            import logging
            from cache_decorator import Cache
            @Cache()
            def test(x):
                return 2 * x
            logger = logging.getLogger("cache.test")
            logger.setLevel(logging.DEBUG)
            ```
        log_format: str = '%(asctime)-15s[%(levelname)s]: %(message)s'
            Formatting of the default logger on stderr. Informations on how the formatting works can be found at
            https://docs.python.org/3/library/logging.html . Moreover, as explained in the log_level, you can get
            a referfence to the logger and fully customize it.
        backup_path: str = None,
            If the serialization fails, the decorator will try to save the computed result as a pickle.
            This parameter is the formatter for the path where to save the backup result.
            If it's None, it will use the same path of the cache and append `_backup.pkl` at the end.
            This will never overwrite any file, so if a file at the current path is present, a random path will be
            generated.
            For this reason in the formatter you can use any variable such as {cache_dir}, {cache_path}, or the arguments
            of the function. Moreover, there is also another two additional parameters, {_date} which is the date of the backup, and {_rnd} which is a random string that will
            guarantee that no file has the same name.
        backup: bool = True,
            If the cache should backup the result to a .pkl in case of exception during the serializzation.
            This flag is mainly for debug pourpouses.
        optional_path_keys: Optional[List[str]] = None
            This argument can be used only if the cache_path is a Dict, otherwise we will raise an
            exception. Otherwise, if the cache_path is a Dict, this is the list of keys that are
            optional to dump and/or load. This is used to cache results with possibly
            missing keys.
        enable_cache_arg_name: Optional[str] = None,
            This paramer specify the name of a boolean argument that, if given
            to the cached function, will enable or disable the caching
            dynamically. If the argument is not passed the cache will be
            enabled by default. This argument, if passed, will automatically be
            added to the `args_to_ignore` so that it doesn't ruin the caching.
            Currently, this can **only** be a kwarg, in future releases I will
            implement it also for args.
        capture_enable_cache_arg_name: bool = True,
            If this parameter is set, the cache will capture the enable cache arg
            and NOT pass it to the decorated function.
        """
        self.log_level = log_level
        self.log_format = log_format
        self.args_to_ignore = list(args_to_ignore)
        self.use_source_code = use_source_code
        self.validity_duration = parse_time(validity_duration)
        self.use_approximated_hash = use_approximated_hash

        self.cache_path = cache_path
        self.is_backup_enabled = backup
        self.backup_path = backup_path
        self.cache_dir = cache_dir or os.environ.get("CACHE_DIR", "./cache")

        self.logger = None
        self.function_info = None
        self.decorated_function = None

        if load_kwargs is None:
            load_kwargs = {}
        if dump_kwargs is None:
            dump_kwargs = {}

        self.load_kwargs, self.dump_kwargs = load_kwargs, dump_kwargs
        self.enable_cache_arg_name = enable_cache_arg_name
        self.capture_enable_cache_arg_name = capture_enable_cache_arg_name

        self.optional_path_keys = optional_path_keys if optional_path_keys is not None else []

        if not self.optional_path_keys and not isinstance(self.cache_path, dict):
            pass # optional_path_keys initialized cleanly to []
        elif self.optional_path_keys and not isinstance(self.cache_path, dict):
            raise ValueError(
                (
                    f"The argument `optional_path_keys` with value '{self.optional_path_keys}' has no "
                    f"meaning if the `cache_path` isn't a dict. `cache_path`='{self.cache_path}'"
                )
            )

        self._check_path_sanity(cache_path)

    def _check_path_sanity(
        self, path: str | tuple[t.Any, ...] | list[t.Any] | dict[str, t.Any]
    ) -> None:
        """Check that at least one backend exists that can handle the given path.
        This is just a quality of life check to raise an exception early and not
        after the computation is done."""
        test_bk = Backend({}, {})
        if isinstance(path, str):
            if not test_bk.support_path(path):
                raise ValueError(
                    (
                        f"There is not backend that can support the path '{path}'. "
                        f"The available extensions are '{test_bk.get_supported_extensions()}'."
                    )
                )
        elif isinstance(path, list) or isinstance(path, tuple):
            for sub_path in path:
                self._check_path_sanity(sub_path)
        elif isinstance(path, dict):
            for _arg, sub_path in path.items():
                self._check_path_sanity(sub_path)

            missing_keys = set(self.optional_path_keys) - set(path.keys())
            if len(missing_keys) != 0:
                raise ValueError(
                    (
                        "The `cache_path` dictionary is missing some keys defined in "
                        f"the `optional_path_keys` arg. Specifically: {missing_keys}"
                    )
                )
        else:
            raise ValueError(
                (
                    f"Sorry, the path '{path}' is not in one of the supported formats."
                    "We support a string, a list, tuple, or dict of paths."
                )
            )

    @staticmethod
    def store(obj: t.Any, path: str) -> None:
        """Store an object at a path, this automatically choose the correct backend.

        Arguments
        ---------
            obj: Object,
                The object to store
            path: str,
                Where to store the file, based on its extension it will choose the correct backend.
        """
        dirname = os.path.dirname(os.path.abspath(path))
        if dirname != "":
            os.makedirs(dirname, exist_ok=True)
        Backend({}, {}).dump(obj, path)

    @staticmethod
    def load(path: str) -> t.Any:
        """
        Load an object from a file, this automatically choose the correct backend.

        Arguments
        ---------
            path: str,
                The path to the file to load file, based on its extension it will choose the correct backend.

        Returns
        -------
        The loaded object.
        """
        return Backend({}, {}).load({}, path)

    @staticmethod
    def compute_path(function: t.Callable[..., t.Any], *args: t.Any, **kwargs: t.Any) -> str:
        """Return the path that a file would have if the given function
        woule be called with the given arguments.

        """
        # If we are dealing with a cached function then unpack it:
        if "__cached_function" not in dir(function):
            raise ValueError(
                "You cannot compuite the path of a function which is not decorated with the Cache decorator."
            )

        instance = getattr(function, "__cacher_instance")
        function = getattr(function, "__cached_function")

        return instance._get_formatted_path(
            args, kwargs, function_info=instance._compute_function_info(function)
        )

    def _compute_function_info(self, function: t.Callable[..., t.Any]) -> dict[str, t.Any]:
        function_args_specs = inspect.getfullargspec(function)

        function_info = {
            # Name of the function
            "function_name": get_function_name(function),
            # Arguments names
            "args": function_args_specs.args or [],
            "defaults": function_args_specs.defaults or [],
            "kwonlydefaults": function_args_specs.kwonlydefaults or {},
            "args_to_ignore": self.args_to_ignore,
        }

        if self.use_source_code:
            # Get the sourcode of the funciton
            # This will be used in the hash so that old
            # Caches will not be loaded
            function_info["source"] = "".join(inspect.getsourcelines(function)[0])

        return function_info

    def _backup(self, result: t.Any, path: t.Any, exception: t.Any, args: tuple[t.Any, ...], kwargs: dict[str, t.Any]) -> Exception:
        """This function handle the backupping of the data when an the serialization fails."""
        assert self.logger is not None
        assert self.function_info is not None

        # Check if it's a structured path
        if isinstance(path, list) or isinstance(path, tuple):
            return self._backup(result, path[0], exception, args, kwargs)

        elif isinstance(path, dict):
            return self._backup(result, next(iter(path.values())), exception, args, kwargs)

        date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        # Ensure that the we won't overwrite anything
        while True:
            # Generate a new random value
            rnd = random_string(
                40
            )  # hardcoded length but if they are enough for git it's enough for us.
            
            if self.backup_path is None:
                self.backup_path = "{cache_path}_backup_{_rnd}.pkl"

            backup_path = self._get_formatted_path(
                args,
                kwargs,
                formatter=self.backup_path,
                extra_kwargs={
                    "_date": date,
                    "_rnd": rnd,
                    "cache_path": self._get_formatted_path(args, kwargs),
                },
            )
            # Check for the existance
            if os.path.exists(backup_path):
                # If the file exists and the rnd var is not used
                # we force it so we don't get stuck in the loop.
                if "{_rnd}" not in self.backup_path:
                    self.backup_path += "{_rnd}"
                continue
            break

        # Inform the user about hte problem
        self.logger.critical(
            "Couldn't save the result of the function '%s'. "
            "Saving the result as a pickle at:\n%s"
            "\nThe file was gonna be written at:\n%s\n",
            self.function_info["function_name"],
            backup_path,
            path,
        )
        # Backup the result
        dirname = os.path.dirname(backup_path)
        if dirname != "":
            os.makedirs(dirname, exist_ok=True)

        with open(backup_path, "wb") as f:
            pickle.dump(result, f)
        # Re-raise the exception
        exception.backup_path = backup_path
        exception.path = path
        exception.result = result
        return exception

    def _get_metadata_path(self, path: str) -> str:
        return path + ".metadata"

    def _is_cache_enabled(self, args: tuple[t.Any, ...], kwargs: dict[str, t.Any], inner_self: t.Any = None) -> tuple[t.Any, tuple[t.Any, ...], dict[str, t.Any]]:
        assert self.function_info is not None
        
        # if enable_cache_arg_name is not defined, then forward
        if self.enable_cache_arg_name is None:
            return True, args, kwargs

        # if the arg is in the kwargs, ez, pop the value and retrun
        if self.enable_cache_arg_name in kwargs:
            if self.capture_enable_cache_arg_name:
                cache_enabled = kwargs.pop(self.enable_cache_arg_name, True)
            else:
                cache_enabled = kwargs[self.enable_cache_arg_name]
            return cache_enabled, args, kwargs

        # Normalize args and kwargs
        params = get_params(self.function_info, args, kwargs)

        # TODO!: implement this
        # # if it was in the args, we need to remove it from there
        # if self.enable_cache_arg_name in params:
        #     cache_enabled = kwargs.pop(self.enable_cache_arg_name, True)
        #     return cache_enabled, args, kwargs

        # Add the self if we are in a method
        if inner_self is not None:
            params["self"] = inner_self

        # otherwise it's either an attribute or method call we should resolve

        # Get the name of the base element and the attributes chain
        root, *attrs = self.enable_cache_arg_name.strip("()").split(".")

        # This means that the parameter was setted but the arg wasn't passed
        # so we should default to true
        if root not in params:
            return True, args, kwargs

        # Get the params to use for the attributes chain
        root = params[root]

        # Follow the attributes chain
        for attr in attrs:
            root = getattr(root, attr)

        # Check if we have to call the function or not
        if (
            inspect.isfunction(root)
            or inspect.ismethod(root)
            or inspect.isbuiltin(root)
        ):
            root = root()

        return root, args, kwargs

    def _load(self, path: t.Any) -> t.Any:
        assert self.logger is not None

        # Check if it's a structured path
        if isinstance(path, list) or isinstance(path, tuple):
            result_list = []
            for p in path:
                loaded_cache = self._load(p)

                if loaded_cache is None:
                    return None

                result_list.append(loaded_cache)

            if isinstance(path, tuple):
                return tuple(result_list)
            return result_list

        elif isinstance(path, dict):
            result_dict = {}
            for key, p in path.items():
                loaded_cache = self._load(p)

                # if we couldn't load the cache
                if loaded_cache is None:
                    # and it's optional it's fine, go on loading the other ones
                    if key in self.optional_path_keys:
                        continue
                    # else it's an error and we cannot load the required data
                    # therefore the cache is invalid
                    else:
                        return None

                result_dict[key] = loaded_cache
            return result_dict

        # Check if the cache exists and is readable
        if not os.path.isfile(path):
            self.logger.info("The cache at path '%s' does not exists.", path)
            return None

        self.logger.info("Loading cache from %s", path)

        metadata_path = self._get_metadata_path(path)

        # Load the metadata if present
        if os.path.isfile(metadata_path):
            self.logger.info("Loading the metadata file at '%s'", metadata_path)
            with open(metadata_path, "r", encoding="utf8") as f:
                metadata = json.load(f)
        else:
            self.logger.info("The metadata file at '%s' do not exists.", metadata_path)
            # TODO: do we need to to more stuff?
            metadata = {}

        # Check if the cache is still valid
        if self.validity_duration is not None:
            enlapsed_time = time() - metadata.get("creation_time", float("-inf"))
            if enlapsed_time > self.validity_duration:
                try:
                    os.remove(path)
                except:
                    pass
                return None

        # actually load the values
        return Backend(self.load_kwargs, self.dump_kwargs).load(
            metadata.get("backend_metadata", {}), path
        )

    def _check_return_type_compatability(self, result: t.Any, path: t.Any) -> None:
        # Check if it's a structured path
        if isinstance(path, list) or isinstance(path, tuple):
            assert isinstance(result, list) or isinstance(result, tuple)
            assert len(result) == len(path)
        elif isinstance(path, dict):
            assert isinstance(result, dict)
            required_keys = set(path.keys()) - set(self.optional_path_keys)
            missing_keys = required_keys - set(result.keys())
            if len(missing_keys) != 0:
                raise ValueError(
                    (
                        f"The result of the cached function has keys '{result.keys()}' that does "
                        f"not match with the required ones '{required_keys}'"
                    )
                )

            extra_keys = set(result.keys()) - set(path.keys())
            if len(extra_keys) != 0:
                raise ValueError(
                    (
                        f"The result of the cached function has keys '{result.keys()}' that does "
                        "not appear in the defined path. In particular the the "
                        f"extra keys are '{extra_keys}'"
                    )
                )
            return

    def _dump(self, args: tuple[t.Any, ...], kwargs: dict[str, t.Any], result: t.Any, path: t.Any, start_time: float, end_time: float) -> None:
        assert self.logger is not None
        assert self.function_info is not None
        assert self.decorated_function is not None

        # Check if it's a structured path
        if isinstance(path, list) or isinstance(path, tuple):
            for r, p in zip(result, path):
                self._dump(args, kwargs, r, p, start_time, end_time)
            return
        elif isinstance(path, dict):
            for key in result.keys():
                self._dump(args, kwargs, result[key], path[key], start_time, end_time)
            return

        # Dump the file
        self.logger.info("Saving the cache at %s", path)
        dirname = os.path.dirname(path)
        if dirname != "":
            os.makedirs(dirname, exist_ok=True)
        dump_start_time = time()
        backend_metadata = (
            Backend(self.load_kwargs, self.dump_kwargs).dump(result, path) or {}
        )
        dump_end_time = time()

        # Compute the metadata
        metadata = {
            # When the cache was created
            "creation_time": start_time,
            "creation_time_human": datetime.fromtimestamp(start_time).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            # How much the function took to compute the result
            "time_delta": end_time - start_time,
            "time_delta_human": humanize.precisedelta(end_time - start_time),
            # How much time it took to serialize the result and save it to a file
            "file_dump_time": dump_end_time - dump_start_time,
            "file_dump_time_human": humanize.precisedelta(
                dump_end_time - dump_start_time
            ),
            # How big is the serialized result
            "file_dump_size": os.path.getsize(path),
            "file_dump_size_human": humanize.naturalsize(os.path.getsize(path)),
            # The arguments used to load and dump the file
            "load_kwargs": self.load_kwargs,
            "dump_kwargs": self.dump_kwargs,
            # Informations about the function
            "function_name": self.function_info["function_name"],
            "function_file": "%s:%s"
            % (
                t.cast(t.Any, self.decorated_function).__code__.co_filename,
                t.cast(t.Any, self.decorated_function).__code__.co_firstlineno,
            ),
            "args_to_ignore": self.function_info["args_to_ignore"],
            "source": self.function_info.get("source", None),
            # The data reserved for the backend to corretly serialize and
            # de-serialize the values
            "backend_metadata": backend_metadata,
        }

        params = {}
        for key, val in get_params(self.function_info, args, kwargs).items():
            if key in self.args_to_ignore:
                continue

            try:
                # Check if it's json serializable
                json.dumps(val)
                params[key] = val
            except:
                pass

        metadata["parameters"] = params

        metadata_path = self._get_metadata_path(path)
        self.logger.info("Saving the cache meta-data at %s", metadata_path)
        with open(metadata_path, "w", encoding="utf8") as f:
            json.dump(metadata, f, indent=4)

    def _decorate_function(self, function: t.Callable[P, R]) -> t.Callable[P, R]:
        # wraps to support pickling
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            assert self.logger is not None
            cache_enabled, args_passed, kwargs_passed = self._is_cache_enabled(args, kwargs)

            # if the cache is not enabled just forward the call
            if not cache_enabled:
                self.logger.info("The cache is disabled")
                result = function(*args_passed, **kwargs_passed)
                self._check_return_type_compatability(result, self.cache_path)
                return result

            # Get the path
            path = self._get_formatted_path(args_passed, kwargs_passed)

            # Try to load the cache
            result = self._load(path)
            # if we got a result, reutrn it
            if result is not None:
                return result

            self.logger.info("Computing the result for %s %s", args_passed, kwargs_passed)
            # otherwise compute the result
            start_time = time()
            result = function(*args_passed, **kwargs_passed)
            end_time = time()

            # Save the result
            try:
                self._check_return_type_compatability(result, path)
                self._dump(args_passed, kwargs_passed, result, path, start_time, end_time)
            except Exception as e:
                if self.is_backup_enabled:
                    raise self._backup(result, path, e, args_passed, kwargs_passed)
                raise e

            return result

        # add a reference to the cached function so we can unpack
        # The caching if needed
        setattr(wrapped, "__cached_function", function)
        setattr(wrapped, "__cacher_instance", self)
        return wrapped

    def _decorate_method(self, function: t.Callable[P, R]) -> t.Callable[P, R]:
        # wraps to support pickling
        @wraps(function)
        def wrapped(inner_self: t.Any, *args: t.Any, **kwargs: t.Any) -> t.Any:
            assert self.logger is not None
            cache_enabled, args_passed, kwargs_passed = self._is_cache_enabled(
                args, kwargs, inner_self=inner_self
            )

            # if the cache is not enabled just forward the call
            if not cache_enabled:
                self.logger.info("The cache is disabled")
                result = t.cast(t.Callable[..., R], function)(inner_self, *args_passed, **kwargs_passed)
                self._check_return_type_compatability(result, self.cache_path)
                return result

            # Get the path
            path = self._get_formatted_path(args_passed, kwargs_passed, inner_self=inner_self)

            # Check that the self is actually hashable
            if not isinstance(inner_self, Hashable):
                raise ValueError(
                    (
                        f"Could not hash self of class `{inner_self.__class__.__name__}` "
                        "because it doesn't implement Hashable (from dict_hash)."
                    )
                )

            # Try to load the cache
            result = self._load(path)
            # if we got a result, reutrn it
            if result is not None:
                return result

            self.logger.info("Computing the result for %s %s", args_passed, kwargs_passed)
            # otherwise compute the result
            start_time = time()
            # Function is a method, but `P.args` matching with explicit self requires careful passing
            # Depending on how it's wrapped, function usually expects self explicitly here
            result = t.cast(t.Callable[..., R], function)(inner_self, *args_passed, **kwargs_passed)
            end_time = time()

            # Save the result
            try:
                self._check_return_type_compatability(result, path)
                self._dump(args_passed, kwargs_passed, result, path, start_time, end_time)
            except Exception as e:
                if self.is_backup_enabled:
                    raise self._backup(result, path, e, args_passed, kwargs_passed)
                raise e

            return result

        # add a reference to the cached function so we can unpack
        # The caching if needed
        setattr(wrapped, "__cached_function", function)
        setattr(wrapped, "__cacher_instance", self)
        return t.cast(t.Callable[P, R], wrapped)

    def _get_formatted_path(
        self,
        args: tuple[t.Any, ...],
        kwargs: dict[str, t.Any],
        formatter: t.Any = None,
        function_info: dict[str, t.Any] | None = None,
        extra_kwargs: dict[str, t.Any] | None = None,
        inner_self: t.Any = None,
    ) -> t.Any:
        """Compute the path adding and computing the needed arguments."""
        assert self.logger is not None
        formatter = formatter or self.cache_path

        if isinstance(formatter, list):
            return [self._get_formatted_path(args, kwargs, f, function_info, extra_kwargs, inner_self) for f in formatter]

        if isinstance(formatter, tuple):
            return tuple([self._get_formatted_path(args, kwargs, f, function_info, extra_kwargs, inner_self) for f in formatter])

        elif isinstance(formatter, dict):
            return {
                key: self._get_formatted_path(args, kwargs, v, function_info, extra_kwargs, inner_self)
                for key, v in formatter.items()
            }

        extra_kwargs = extra_kwargs or {}

        function_info = function_info or self.function_info
        assert function_info is not None
        
        params = get_params(function_info, args, kwargs)
        groups = get_format_groups(formatter)
        groups_set = {match.str_match for match in groups}

        if "_hash" in groups_set:
            data = {"params": params, "function_info": function_info}

            if inner_self is not None:
                data["self"] = inner_self

            params["_hash"] = sha256(data, use_approximation=self.use_approximated_hash)

        self.logger.debug("Got parameters %s", params)

        cache_dir = self.cache_dir
        if formatter != self.cache_dir:
            cache_dir = self._get_formatted_path(
                args, kwargs, self.cache_dir, function_info, extra_kwargs, inner_self
            )

        format_args = {
            **params,
            **function_info,
            **extra_kwargs,
            "cache_dir": cache_dir,
        }

        new_formatter = ""
        old_formatter = formatter
        # Handle the composite paths
        while len(old_formatter) != 0:
            new_match, formatter_remainder = get_next_format_group(old_formatter)

            # there are no more matches just append the remainder
            if new_match is None:
                new_formatter += formatter_remainder
                break

            # check if we should call the value or not
            if new_match.str_match.endswith("()"):
                new_match.str_match = new_match.str_match[:-2]
                # Get the name of the base element and the attributes chain
                root_name, *attrs = new_match.str_match.split(".")
                # Get the params to use for the attributes chain
                root = format_args[root_name]

                # Follow the attributes chain
                for attr in attrs:
                    root = getattr(root, attr)

                # Check if we have to call the function or not
                if (
                    inspect.isfunction(root)
                    or inspect.ismethod(root)
                    or inspect.isbuiltin(root)
                ):
                    root = root()

                sub = str(root)
            else:
                sub = "{" + new_match.str_match + "}"

            new_formatter += old_formatter[: new_match.start]
            new_formatter += sub
            old_formatter = formatter_remainder

        path = new_formatter.format(
            **format_args,
        )
        self.logger.debug("Calculated path %s", path)
        return path

    def _fix_docs(self, function: t.Callable[P, R], wrapped: t.Callable[P, R]) -> t.Callable[P, R]:
        # Copy the doc of decoreated function
        wrapped.__doc__ = function.__doc__
        # Copy the name of the function and add the suffix _cached
        wrapped.__name__ = get_function_name(function) + "_cached"
        return wrapped

    def decorate(self, function: t.Callable[P, R]) -> t.Callable[P, R]:
        """Decorate the function with the cache."""
        self.function_info = self._compute_function_info(function)
        self.decorated_function = function

        if inspect.ismethod(function):
            wrapped = self._decorate_method(function)
        else:
            wrapped = self._decorate_function(function)

        wrapped = self._fix_docs(t.cast(t.Callable[P, R], function), t.cast(t.Callable[P, R], wrapped))
        return wrapped

    def __call__(self, function: t.Callable[P, R]) -> t.Callable[P, R]:
        self.logger = logging.getLogger(__name__ + "." + get_function_name(function))
        # Do not re-initialize loggers if we have to cache multiple functions
        # with the same name
        if not self.logger.hasHandlers():
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter(self.log_format))
            self.logger.addHandler(handler)

        if self.log_level.lower() not in log_levels:
            raise ValueError(
                (
                    f"The logger level {self.log_level.lower()} is not a supported one. "
                    f"The available ones are {list(log_levels.keys())}"
                )
            )
        self.logger.setLevel(log_levels[self.log_level.lower()])

        return self.decorate(function)
