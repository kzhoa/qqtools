import qqtools.utils.qlist as qlist


def test_filter_returns_matching_items():
    data = [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}, {"id": 1, "name": "c"}]
    result = qlist.filter(data, "id", 1)
    assert result == [{"id": 1, "name": "a"}, {"id": 1, "name": "c"}]


def test_filter_returns_none_when_not_found():
    data = [{"id": 1}, {"id": 2}]
    assert qlist.filter(data, "id", 3) is None


def test_find_returns_first_match():
    data = [{"id": 1}, {"id": 2}, {"id": 2, "name": "x"}]
    assert qlist.find(data, "id", 2) == {"id": 2}


def test_find_returns_none_when_not_found():
    data = [{"id": 1}, {"id": 2}]
    assert qlist.find(data, "id", 9) is None
