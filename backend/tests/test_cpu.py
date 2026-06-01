from backend.app.cotail.cpu import format_cpu_range, parse_cpu_range


def test_cpu_range_roundtrip():
    assert parse_cpu_range("0-2,8,10-11") == [0, 1, 2, 8, 10, 11]
    assert format_cpu_range([0, 1, 2, 8, 10, 11]) == "0-2,8,10-11"

