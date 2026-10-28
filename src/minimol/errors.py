"""Stable error types. Messages are English so tests and logs stay grep-friendly."""


class MinimolError(Exception):
    """Base error for expected, user-facing failures."""


class LicenseError(MinimolError):
    """A source is not approved for the requested use."""


class SplitLeakageError(MinimolError):
    """A split, teacher cache, or scaler would leak held-out labels."""


class RemoteCodeNotReviewed(MinimolError):
    """trust_remote_code was requested before a pinned revision was reviewed."""


class AlignmentError(MinimolError):
    """Token coordinates do not match the nucleotide sequence."""
