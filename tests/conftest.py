import os

import pytest

os.environ["HOUSING_SYNTHETIC"] = "1"  # tests never touch the real (uncommitted) data


@pytest.fixture(scope="session")
def built():
    """Write the synthetic raw files and build the processed tables once per test session."""
    from housing import panel, synthetic
    from housing.config import data_dirs

    raw, out = data_dirs()
    synthetic.write_raw(raw)
    return panel.build(raw, out)
