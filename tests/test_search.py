import compass2d.compass
import torch

import garimpo_uncertainty.compass_j

BETA_RANGE, THRESHOLD = 1000.0, 1e-15


def test_lockstep_search_matches_compass2d_to_the_bit_with_fewer_decodes() -> None:
    torch.manual_seed(0)
    decoder = torch.nn.Sequential(
        torch.nn.Conv2d(8, 8, 3, padding=1),
        torch.nn.BatchNorm2d(8),
        torch.nn.ReLU(),
        torch.nn.Conv2d(8, 1, 3, padding=1),
    ).eval()
    decodes = 0

    def decode(z: torch.Tensor) -> torch.Tensor:
        nonlocal decodes
        decodes += 1
        return decoder(z)

    z = torch.randn(6, 8, 16, 16)
    delta = compass2d.compass.channel_delta(torch.randn(6, 8))
    space = garimpo_uncertainty.compass_j.Space(
        decode, garimpo_uncertainty.compass_j.soft_area(10.0), z, delta
    )
    with torch.no_grad():
        start = space.shifted(torch.zeros(6, dtype=torch.float64))
    # Targets above, below and at the unshifted area, and one past every reachable area.
    target = start * torch.tensor([1.3, 0.7, 1.0, 1.05, 0.2, 1e6], dtype=torch.float64)
    searched = (decode, space.metric, z, delta, target, BETA_RANGE, THRESHOLD)

    decodes = 0
    symmetric = compass2d.compass.binary_search_scores(*searched)
    lower, upper = compass2d.compass.asymmetric_binary_search_scores(*searched)
    separate = decodes
    decodes = 0
    found = garimpo_uncertainty.compass_j.search(space, target, BETA_RANGE, THRESHOLD)

    for (score, saturated), (expected, expected_saturated) in zip(
        found, [symmetric, lower, upper], strict=True
    ):
        assert torch.equal(score, expected)
        assert torch.equal(saturated, expected_saturated)
    assert decodes < separate * 0.7


def test_memo_decodes_a_nan_beta_once() -> None:
    z = torch.zeros(2, 1, 2, 2)
    space = garimpo_uncertainty.compass_j.Space(
        lambda shifted: shifted, lambda out: out.flatten(1).sum(1).double(), z, torch.ones_like(z)
    )
    memo = garimpo_uncertainty.compass_j.Memo(space)
    areas = memo(torch.tensor([[float("nan"), 1.0], [0.5, 1.0]], dtype=torch.float64))
    assert memo.decodes == 2
    expected = torch.tensor([[float("nan"), 4.0], [2.0, 4.0]], dtype=torch.float64)
    torch.testing.assert_close(areas, expected, rtol=0, atol=0, equal_nan=True)
