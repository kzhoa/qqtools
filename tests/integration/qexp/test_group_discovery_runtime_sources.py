"""Runtime extraction works without the development scripts package."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.integration


def test_runtime_source_projection_and_resume_without_scripts(tmp_path):
    source = tmp_path / "submission.json"
    source.write_text(
        json.dumps(
            {
                "meta": {"schema_version": 6},
                "submission": {
                    "operation_id": "op-runtime",
                    "state": "committed",
                    "target_group": "runtime-group",
                    "resolved_context": {"task_ids": ["task-a", "task-b"]},
                    "commit_plan": {"group_membership_sequences": [1, 2]},
                },
            },
            ensure_ascii=False,
        )
    )
    program = textwrap.dedent(
        """
        import importlib.abc
        import json
        import sys
        from pathlib import Path

        class RejectDevelopmentScripts(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'scripts' or fullname.startswith('scripts.'):
                    raise AssertionError('runtime imported development code: ' + fullname)

        sys.meta_path.insert(0, RejectDevelopmentScripts())
        from qqtools.plugins.qexp.runtime.group_discovery.checkpoint import (
            _encode_json, _validate_cross_consistency,
        )
        from qqtools.plugins.qexp.runtime.group_discovery.json_stream import Scanner
        from qqtools.plugins.qexp.runtime.group_discovery.source_revision import BoundSource
        from qqtools.plugins.qexp.runtime.group_discovery.submission_projection import SubmissionProjection

        path = Path(sys.argv[1])
        chunks = []
        ends = []
        projection = SubmissionProjection('op-runtime', 'runtime-group', chunks.append, ends.append)
        source = BoundSource.open(path)
        try:
            scanner = Scanner(source, lambda span: None, chunk_bytes=7, emit_bytes=projection.feed)
            for _ in range(12):
                scanner.step(13, max_fragments=2)
            lexer_state = scanner.snapshot()
            parser_state = projection.snapshot()
            _validate_cross_consistency(lexer_state, parser_state, source.revision)
            saved = json.loads(_encode_json({'lexer': lexer_state, 'parser': parser_state}))
            revision = source.revision
        finally:
            source.close()
        source = BoundSource.open(path, expected_revision=revision)
        try:
            source.seek(saved['lexer']['offset'])
            projection = SubmissionProjection.from_snapshot(
                'op-runtime', 'runtime-group', chunks.append, ends.append, saved['parser'],
            )
            scanner = Scanner.from_snapshot(
                source, lambda span: None, saved['lexer'], emit_bytes=projection.feed,
            )
            for _ in range(1000):
                if scanner.step(13, max_fragments=2).is_complete:
                    break
            else:
                raise AssertionError('runtime source projection did not complete')
            summary = projection.finish()
            source.verify()
            _validate_cross_consistency(scanner.snapshot(), projection.snapshot(), source.revision)
            values = {}
            for chunk in chunks:
                key = f'{chunk.kind}:{chunk.ordinal}'
                values.setdefault(key, bytearray()).extend(chunk.data)
            print(json.dumps({
                'schema': projection.source_schema_version,
                'values': {key: bytes(value).decode() for key, value in values.items()},
                'ends': len(ends),
                'summary': repr(summary),
                'scripts_loaded': any(name == 'scripts' or name.startswith('scripts.') for name in sys.modules),
            }))
        finally:
            source.close()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(source)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["schema"] == 6
    assert observed["ends"] == 4
    assert sorted(observed["values"].values()) == ["1", "2", "task-a", "task-b"]
    assert not observed["scripts_loaded"]
