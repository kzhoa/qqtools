import re

from qqtools.config.qtime import date_str, hms_form, now_str


def test_now_str_format():
    value = now_str()
    assert re.fullmatch(r"\d{8}_\d{6}", value)


def test_date_str_format():
    value = date_str()
    assert re.fullmatch(r"\d{4}", value)


def test_hms_form_compact():
    assert hms_form(0) == "0s"
    assert hms_form(61) == "1min 1s"
    assert hms_form(3600) == "1h"


def test_hms_form_full_format():
    assert hms_form(3661, fullFormat=True) == "01h 01min 01s"
