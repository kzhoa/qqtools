import datetime
import time

__all__ = ["now_str", "date_str", "hms_form"]


def now_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def date_str():
    return datetime.datetime.now().strftime("%m%d")


def hms_form(sec):
    return time.strftime("%Hh %Mmin %Ss", time.gmtime(sec))
