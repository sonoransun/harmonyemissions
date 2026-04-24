"""Tests for backend routing and availability."""

import pytest

from harmonyemissions import Laser, Target, simulate
from harmonyemissions.backends.epoch import EpochNotAvailable
from harmonyemissions.backends.smilei import SmileiNotAvailable


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        simulate(Laser(a0=1.0), Target.overdense(100), model="rom", backend="notAthing")


def test_smilei_backend_clearly_reports_missing_executable():
    # SMILEI almost certainly not installed in test environments.
    with pytest.raises((SmileiNotAvailable, NotImplementedError)):
        simulate(
            Laser(a0=1.0), Target.overdense(100), model="rom", backend="smilei",
            executable="definitely-not-smilei",
        )


def test_epoch_backend_clearly_reports_missing_executable():
    with pytest.raises((EpochNotAvailable, NotImplementedError)):
        simulate(
            Laser(a0=1.0), Target.overdense(100), model="rom", backend="epoch",
            executable="definitely-not-epoch",
        )


def test_smilei_rejects_unsupported_model():
    # Falls through the shutil.which check with a real missing exe.
    with pytest.raises(SmileiNotAvailable):
        simulate(
            Laser(a0=0.1), Target.gas("Ar"), model="lewenstein", backend="smilei",
            executable="not-a-real-thing",
        )
