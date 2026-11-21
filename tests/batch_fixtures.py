"""Synthetic market loader for batch tests (imported by name in the child process)."""
from tests.helpers import synth


def load():
    return synth(T=90, seed=21, dollar_volume=2e6)


def load_other():
    return synth(T=90, seed=22, dollar_volume=2e6)
