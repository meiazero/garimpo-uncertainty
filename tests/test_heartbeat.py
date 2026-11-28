import sys
import time

import garimpo_uncertainty.pipeline.run


def test_a_running_stage_logs_that_it_is_alive_and_stops_when_it_ends(tmp_path, monkeypatch):
    run = garimpo_uncertainty.pipeline.run
    monkeypatch.setattr(run, "HEARTBEAT_SECONDS", 0.2)
    monkeypatch.setattr(run, "gpu_use", lambda: " · GPU 97 %, 5731 MiB, 23034 MiB")
    log = run.Log(tmp_path / "pilot-model.log", total=1)
    assert log.run("train", [sys.executable, "-c", "import time; time.sleep(1.5)"]) == 0
    ended = log.path.read_text(encoding="utf-8")
    time.sleep(0.6)
    lines = ended.splitlines()
    alive = [i for i, line in enumerate(lines) if " === alive train: " in line]
    assert len(alive) >= 3, lines
    assert all(lines[i].endswith(" · GPU 97 %, 5731 MiB, 23034 MiB") for i in alive), lines
    assert alive[-1] < next(i for i, line in enumerate(lines) if "=== end train" in line)
    assert log.path.read_text(encoding="utf-8") == ended, "no alive line may follow the end"
