import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "colab_session.py"


def test_a_session_step_logs_that_it_is_alive_and_stops_when_it_ends(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("colab_session", SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    monkeypatch.setattr(script, "SESSION_LOG", tmp_path / "session.log")
    monkeypatch.setattr(script, "CODE", tmp_path / "absent")
    monkeypatch.setattr(script, "ALIVE_EVERY", 0.2)
    session = script.Session()
    assert session.ran(sys.executable, "-c", "import time; time.sleep(1.5)")
    session.say("next step")
    # A quiet step is a copy to Drive, which writes no alive lines.
    assert session.ran(sys.executable, "-c", "import time; time.sleep(0.6)", quiet=True)
    lines = script.SESSION_LOG.read_text(encoding="utf-8").splitlines()
    alive = [i for i, line in enumerate(lines) if " === alive " in line]
    assert len(alive) >= 3, lines
    assert all(" GiB" in lines[i] for i in alive), lines
    assert alive[-1] < next(i for i, line in enumerate(lines) if line.endswith(" next step")), lines
