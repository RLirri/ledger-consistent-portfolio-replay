"""Typed failures. A run that raises any of these has no scientifically valid result."""


class ScenarioInputError(ValueError):
    """Invalid configuration, market data, features or targets detected before or during a run."""


class RunStateError(AssertionError):
    """The ledger reached a state the configuration does not permit (negative cash, short, non-finite value)."""


class RunValidationError(RuntimeError):
    """A completed run failed post-run scientific validation."""
