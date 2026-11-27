"""How the figures write numbers: as the dissertation's text does."""

import matplotlib.pyplot as plt
import pytest

from garimpo_uncertainty.figures import common


@pytest.mark.parametrize(
    ("value", "digits", "text"),
    [
        (16048, 0, "16.048"),
        (1047.87, 2, "1.047,87"),
        (-0.05, 2, "\N{MINUS SIGN}0,05"),
        (999.6, 0, "1.000"),
        (999, 0, "999"),
    ],
)
def test_a_number_marks_thousands_with_a_dot_and_decimals_with_a_comma(
    value: float, digits: int, text: str
) -> None:
    assert common.number(value, digits) == text


def tick_labels(low: float, high: float, ticks: list[float]) -> tuple[list[str], str]:
    fig, ax = plt.subplots()
    ax.set_xlim(low, high)
    ax.set_xticks(ticks)
    common.decimal_comma(fig)
    fig.canvas.draw()
    labels = [label.get_text() for label in ax.get_xticklabels()]
    offset = ax.xaxis.get_major_formatter().get_offset()
    plt.close(fig)
    return labels, offset


def test_tick_labels_mark_thousands_and_decimals_as_the_text_does() -> None:
    assert tick_labels(0, 16000, [0, 15000])[0] == ["0", "15.000"]
    assert tick_labels(0, 1, [0, 0.5])[0] == ["0,0", "0,5"]
    assert tick_labels(1000, 1001, [1000.5])[0] == ["1.000,5"]
    assert tick_labels(-2000, 0, [-1500])[0] == ["\N{MINUS SIGN}1.500"]


def test_an_offset_axis_keeps_its_ticks_relative_to_the_offset() -> None:
    labels, offset = tick_labels(1_000_000, 1_000_400, [1_000_100, 1_000_200])
    assert offset
    assert labels == ["100", "200"]
