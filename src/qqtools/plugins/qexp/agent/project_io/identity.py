"""Exact identity of a validated Project binding in an executor epoch."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValidationIdentity:
    runtime_id: str
    executor_epoch: str
    project_id: str
    shared_root: str
    machine_name: str
    registration_generation: str
    runtime_root: str
    registry_revision: int

    @property
    def request_owner(self) -> tuple[str, str, str]:
        return (self.runtime_id, self.project_id, self.registration_generation)
