import pytest


@pytest.mark.parametrize("run", ["action.json"], indirect=True)
def test_action(run):
    assert run == 0


@pytest.mark.parametrize("plot", ["action.json"], indirect=True)
def test_plot_action(plot):
    assert plot == 0
