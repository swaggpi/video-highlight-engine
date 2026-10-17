from vhe.timing import Counter, hms


def test_counter_throttles_and_reports_eta():
    msgs = []
    c = Counter(msgs.append, "embedding frames", every_s=60)
    c(10, 100)
    c(20, 100)  # within the throttle window: dropped
    c(100, 100)  # completion always reported
    assert len(msgs) == 2
    assert msgs[0].startswith("embedding frames 10/100 (10%)") and "eta" in msgs[0]
    assert msgs[1] == "embedding frames 100/100 (100%)"


def test_hms():
    assert hms(75) == "1m15s" and hms(3725) == "1h02m"
