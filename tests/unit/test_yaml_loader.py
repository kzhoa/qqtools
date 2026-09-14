import pytest
import yaml

from qqtools.config.yaml import QExpandSafeLoader, load_yaml


def test_qexpand_safe_loader_supports_dumped_python_scalars():
    data = yaml.load(
        "shape: !!python/tuple\n- 3\n- 224\n- 224\nvalue: !!python/complex '1+2j'\n",
        Loader=QExpandSafeLoader,
    )

    assert data["shape"] == (3, 224, 224)
    assert data["value"] == 1 + 2j


def test_load_yaml_without_inheritance_uses_expanded_safe_loader(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("shape: !!python/tuple\n- 3\n- 224\n- 224\n")

    assert load_yaml(path, inherit=False).shape == (3, 224, 224)


def test_qexpand_safe_loader_supports_imported_python_names():
    data = yaml.load("value: !!python/name:builtins.len ''\n", Loader=QExpandSafeLoader)

    assert data["value"] is len


def test_qexpand_safe_loader_does_not_construct_python_objects():
    with pytest.raises(yaml.constructor.ConstructorError):
        yaml.load(
            "value: !!python/object/apply:os.system ['echo unsafe']\n",
            Loader=QExpandSafeLoader,
        )
