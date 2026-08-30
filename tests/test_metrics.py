"""Unit tests for callosum.training._metrics -- pure Python, no torch needed,
so unlike most of tests/ these run in CI as well as locally."""

from callosum.training._metrics import MetricLogger


class _FakeWriter:
    def __init__(self):
        self.calls = []

    def add_scalar(self, tag, value, step):
        self.calls.append((tag, value, step))


def test_log_reaches_the_writer_unchanged() -> None:
    w = _FakeWriter()
    m = MetricLogger(w)
    m.log("train/return", 1.5, 100)
    assert w.calls == [("train/return", 1.5, 100)]


def test_works_without_a_writer() -> None:
    m = MetricLogger()  # None writer -- the test/offline path
    m.log("train/return", 1.5, 100)
    assert "train/return" in m.summary()


def test_coerces_tensor_like_values() -> None:
    class _Scalar:
        def item(self):
            return 2.5

    m = MetricLogger()
    m.log("losses/jepa_loss", _Scalar(), 1)
    assert "2.5" in m.summary()


def test_headline_survives_iterations_without_episodes() -> None:
    """Episode metrics only appear when an episode finished; the bar must keep
    showing the last known value instead of blanking."""
    m = MetricLogger()
    m.log("train/return", 3.0, 1)
    m.iteration_line(1, 10, 100, 500)
    assert m.headline()["return"] == "3"


def test_iteration_line_mentions_progress_and_headline() -> None:
    m = MetricLogger()
    m.log("train/return", -40.5, 12800)
    line = m.iteration_line(3, 78, 12800, 4600)
    assert "iter 3/78" in line
    assert "step 12800" in line
    assert "return -40.5" in line


def test_summary_reports_first_last_min_max() -> None:
    m = MetricLogger()
    for v in (1.0, 5.0, 3.0):
        m.log("train/return", v, 0)
    out = m.summary()
    assert "train/return" in out
    assert "1" in out and "5" in out and "3" in out


def test_summary_is_safe_when_nothing_was_logged() -> None:
    assert MetricLogger().summary() == "no metrics recorded"
