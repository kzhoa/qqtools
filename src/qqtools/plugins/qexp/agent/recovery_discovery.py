"""Read-only qualification for routine machine recovery checks."""

from .context import MachineRuntime, ProjectBinding
from .recovery_capture import recovery_owner, recovery_source


def read_binding_capture(runtime: MachineRuntime, binding: ProjectBinding) -> dict | None:
    from ..runtime.responsibility_qualification import read_discovery_completion
    from ..runtime.responsibility_store import Unavailable

    root = runtime.project_paths(binding.project_id)["root"]
    proof = read_discovery_completion(root, recovery_owner(runtime, binding))
    if proof is not None:
        source = recovery_source(runtime, binding)
        if proof["legacy_source"] != (str(source) if source is not None else None):
            raise Unavailable("discovery capture source changed")
    return proof
