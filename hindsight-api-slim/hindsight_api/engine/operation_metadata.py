"""
Typed metadata models for async operations.

These dataclasses define the structure of result_metadata for different operation types.
The metadata is exposed in the API for debugging purposes and may change without notice.
"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class BatchRetainParentMetadata:
    """Metadata for parent batch_retain operations (when split into sub-batches)."""

    items_count: int
    total_tokens: int
    num_sub_batches: int
    is_parent: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)


@dataclass
class BatchRetainChildMetadata:
    """Metadata for child batch_retain operations (individual sub-batches)."""

    items_count: int
    parent_operation_id: str
    sub_batch_index: int
    total_sub_batches: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)


@dataclass
class RetainMetadata:
    """Metadata for regular retain operations (non-batched, deprecated async path)."""

    items_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)


@dataclass
class ConsolidationMetadata:
    """Metadata for consolidation operations."""

    # Currently empty, but structure for future fields
    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)


@dataclass
class RefreshMentalModelMetadata:
    """Metadata for mental model refresh operations."""

    mental_model_id: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for JSON serialization."""
        return asdict(self)


def describe_failed_children(children: list[dict[str, Any]]) -> str:
    """One clause per failed sub-batch for the parent's ``error_message``.

    Format: ``sub-batch 2/2 (<child operation_id>) failed: <child error_message>``,
    clauses joined by ``; ``. Each entry needs ``operation_id``, ``error_message``
    and the parsed ``result_metadata`` (``BatchRetainChildMetadata`` fields).
    Shared by the engine and the poller, which both finalise parents. Before
    2026-10-02 the parent only said "One or more sub-batches failed"; the cause
    (op caaef926: "Anthropic returned invalid JSON after 11 attempts") was
    visible on the child row alone.
    """
    clauses = []
    for child in children:
        metadata = child.get("result_metadata") or {}
        clauses.append(
            f"sub-batch {metadata.get('sub_batch_index', '?')}/{metadata.get('total_sub_batches', '?')} "
            f"({child['operation_id']}) failed: {child.get('error_message') or 'no error_message'}"
        )
    return "; ".join(clauses)
