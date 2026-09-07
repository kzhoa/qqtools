def test_qimport_star_export_names_resolve():
    namespace = {}
    exec("from qqtools.qimport import *", namespace)

    assert {"LazyImport", "LazyImportErrorProxy", "is_imported", "import_common"} <= namespace.keys()
