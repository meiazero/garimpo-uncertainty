from tests.conftest import cpu_list


def test_cpu_list_reads_ranges_and_single_cpus() -> None:
    assert cpu_list("0,2-3\n") == {0, 2, 3}
