# NOTE: This test is intentionally minimal.
#
# funmodel-dwpose ships as a `funmodel.dwpose` sub-package layered on top of
# the base `funmodel` distribution (both publish files under the shared
# `funmodel/` top-level namespace). Importing it pulls in a heavy dependency
# chain (torch, onnxruntime, opencv, fundrive, funget). No network/filesystem
# side effects happen at import time -- `DWposePredict.load()` only performs
# downloads when the class is instantiated, which this smoke test avoids.
import funmodel.dwpose
from funmodel.dwpose import DWposePredict


def test_import_funmodel_dwpose():
    assert funmodel.dwpose is not None


def test_import_dwpose_predict_class():
    assert DWposePredict is not None
