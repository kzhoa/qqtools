import datetime
import time

__all__ = ["now_str", "date_str", "hms_form"]


def now_str():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def date_str():
    return datetime.datetime.now().strftime("%m%d")


def hms_form(sec, fullFormat=False):
    if fullFormat:
        return time.strftime("%Hh %Mmin %Ss", time.gmtime(sec))

    t = time.gmtime(sec)
    h = f"{t.tm_hour}h" if t.tm_hour else ""
    m = f"{t.tm_min}min" if t.tm_min else ""
    s = f"{t.tm_sec}s" if t.tm_sec or not (h or m) else ""
    return " ".join(filter(None, [h, m, s]))
