
from main import*
from find_obj import*
import pytest


def test_if_camera_opened() -> None:
    assert cap.isOpened() == False, "[ERROR] - Camera failed to open"
