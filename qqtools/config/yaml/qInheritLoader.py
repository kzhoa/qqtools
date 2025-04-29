from pathlib import Path

import qqtools as qt
import yaml


def python_tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))


def python_set_constructor(loader, node):
    return set(loader.construct_sequence(node))


class QExpandSafeLoader(yaml.SafeLoader):
    def __init__(self, stream):
        super().__init__(stream)
        self.add_constructor("tag:yaml.org,2002:python/tuple", python_tuple_constructor)
        self.add_constructor("tag:yaml.org,2002:python/set", python_set_constructor)


class InheritLoader(QExpandSafeLoader):
    """支持 BASE 继承的 YAML Loader

    **usage:**

    .. code-block:: yaml
        :linenos:
        # simple usage, single path string
        BASE: ./aa.yaml  # 支持相对路径


    .. code-block:: yaml
        :linenos:
        # list usage, multiple path strings
        BASE:
            - ./aa.yaml
            - ../configs/bb.yaml
            - /home/user/cc.yaml


    .. code-block:: yaml
        :linenos:
        BASE:
            optim: ./aa.yaml
            model: ./bb.yaml


    .. code-block:: yaml
        :linenos:
        # dict + list mixing
        BASE:
            optim:
                - ./adam.yaml
                - ./cosine_anneal.yaml
            model:
                - ../config/mymodel.yaml


    .. code-block:: yaml
        :linenos:
        # mixing usage
        BASE:
            - ./base_config.yaml
            - model:
                - ./model_config.yaml
            - optim: ./optim.yaml

    """

    def __init__(self, stream, inherit_depth=0, loaded_files=None):
        super().__init__(stream)
        self._loaded_files = loaded_files or set()  # prevent loop import
        self._inherit_depth = inherit_depth
        self.name = getattr(stream, "name", None)  # read open file name

    def get_single_data(self):
        """qq:
        Hijack YAML Loader
        """
        data = super().get_single_data()
        return self._deep_merge_inherited(data)

    def _deep_merge_inherited(self, data):
        """qq:
        Handle the 'BASE' keyword inheritance by:
        1. Extracting and removing 'BASE' from input data
        2. Loading and merging base file(s) recursively
        3. Updating the base data with current data (current overrides base)
        """
        if "BASE" in data:
            base_files = data.pop("BASE")
            base_data = self._handle_inheritance(base_files)
            return qt.qDict(base_data).recursive_update(data).to_dict()
        return data

    def _handle_inheritance(self, base_files):
        """
        track inherit depth
        """
        if self._inherit_depth >= 5:
            raise yaml.constructor.ConstructorError(None, None, "inherit depth should be <5", None)
        self._inherit_depth += 1

        return self._merge_base_files(base_files)

    def _merge_base_files(self, base_files):
        """qq:
        handle different input types
        """
        # exit
        if isinstance(base_files, str):
            return self._load_base_file(base_files)

        # unroll &  recursive
        if isinstance(base_files, (list, tuple)):
            merged = qt.qDict()
            for base_file in base_files:
                merged.recursive_update(self._merge_base_files(base_file))
            return merged.to_dict()
        elif isinstance(base_files, dict):
            merged = qt.qDict()
            for k, _bfiles in base_files.items():
                _tmp = {k: self._merge_base_files(_bfiles)}
                merged.recursive_update(_tmp)
            return merged
        else:
            raise TypeError(f"BASE must be one of `(str, list, dict)`, not {type(base_files)}")

    def _load_base_file(self, path: str):
        """qq:
        Load and parse a base YAML file with inheritance handling.

        Tracks loaded files to prevent circular references.
        Expand `BASE` keyword if needed.

        Args:
            path (str): Relative path to the base YAML file.

        Returns:
            dict: Parsed YAML content as a dictionary.

        Raises:
            ConstructorError: If circular inheritance is detected or if the file content
                             is not a dictionary.
        """
        base_dir = Path(self.name).resolve().parent
        abs_path = Path(base_dir, path).resolve()
        if abs_path in self._loaded_files:
            raise yaml.constructor.ConstructorError(None, None, f"Loop Inherit: {abs_path}", None)
        self._loaded_files.add(abs_path)

        with open(abs_path, "r") as f:
            base_data = yaml.load(
                f,
                lambda f: InheritLoader(f, inherit_depth=self._inherit_depth, loaded_files=self._loaded_files),
            )
            if not isinstance(base_data, dict):
                raise yaml.constructor.ConstructorError(
                    None, None, f"BASE yaml data must be dict, encountered {type(base_data)}: {abs_path}", None
                )
            return base_data
