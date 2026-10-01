"""Regression checks for audit provenance and immutable historical replay."""

import ast
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from audit_support import historical_fixture, library_provenance


class AuditSupportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / "site-packages" / "autolens"
        self.package.mkdir(parents=True)
        self.init = self.package / "__init__.py"
        self.init.write_text("VERSION = 1\n")
        self.module = SimpleNamespace(__name__="autolens", __file__=str(self.init))
        self.distribution = SimpleNamespace(
            version="1.2.3",
            files=[Path("autolens/__init__.py")],
            locate_file=lambda file: self.package.parent / file,
        )

    def test_installed_identity_changes_when_code_or_version_changes(self):
        with patch(
            "audit_support.importlib.metadata.distribution",
            return_value=self.distribution,
        ):
            first = library_provenance(self.module)
            self.assertEqual(first["kind"], "installed")
            self.assertIsNone(first["sha"])
            self.assertEqual(first, library_provenance(self.module))
            self.init.write_text("VERSION = 2\n")
            self.assertNotEqual(first, library_provenance(self.module))
            self.init.write_text("VERSION = 1\n")
            self.distribution.version = "1.2.4"
            self.assertNotEqual(first, library_provenance(self.module))

    def test_bytecode_cache_does_not_change_installed_identity(self):
        with patch(
            "audit_support.importlib.metadata.distribution",
            return_value=self.distribution,
        ):
            first = library_provenance(self.module)
            cache = self.package / "__pycache__"
            cache.mkdir()
            (cache / "__init__.pyc").write_bytes(b"cache")
            self.assertEqual(first, library_provenance(self.module))

    def test_enclosing_repository_is_not_library_provenance(self):
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        with patch(
            "audit_support.importlib.metadata.distribution",
            return_value=self.distribution,
        ):
            self.assertEqual(library_provenance(self.module)["kind"], "installed")

    def test_unrelated_metadata_is_rejected(self):
        self.distribution.locate_file = lambda file: self.root / "other" / file
        with patch(
            "audit_support.importlib.metadata.distribution",
            return_value=self.distribution,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "No matching installed distribution"
            ):
                library_provenance(self.module)

    def test_checkout_keeps_commit_and_dirty_status(self):
        directory = self.package.parent
        subprocess.run(["git", "init", "-q", str(directory)], check=True)
        subprocess.run(["git", "-C", str(directory), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(directory),
                "-c",
                "user.name=Audit test",
                "-c",
                "user.email=audit@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )
        first = library_provenance(self.module)
        self.assertEqual(first["kind"], "checkout")
        self.assertEqual(len(first["sha"]), 40)
        self.assertEqual(first["dirty"], "")
        self.init.write_text("VERSION = 2\n")
        self.assertTrue(library_provenance(self.module)["dirty"])

    def test_historical_fixture_has_original_method_and_pins(self):
        fixture = historical_fixture()
        self.assertEqual(len(fixture["boundaries"]), 12)
        self.assertEqual(
            fixture["method_source"]["sha"], "8d7747d8ee5486fcbe9da0f137dc2f94640ea332"
        )
        (method,) = ast.parse(fixture["source"]).body
        self.assertIsInstance(method, ast.FunctionDef)
        self.assertEqual(method.name, "hessian_from")
        self.assertFalse(method.decorator_list)

    def test_tampered_source_and_metadata_are_rejected(self):
        fixture = historical_fixture()
        for key in ("source", "method_source"):
            changed = dict(fixture)
            changed[key] = "tampered"
            path = self.root / "tampered.json"
            path.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                historical_fixture(path)

    def test_history_rejects_empty_comparison_evidence(self):
        from history_audit import measure

        path = self.root / "empty.json"
        path.write_text('{"rows": []}')
        with self.assertRaisesRegex(ValueError, "nonempty NumPy evidence"):
            measure(path)

    def test_resume_rejects_different_library_pins_before_solving(self):
        from error_audit import run

        path = self.root / "resume.json"
        path.write_text('{"provenance": {}, "rows": []}')
        with patch.dict("os.environ", {"PYAUTO_SMALL_DATASETS": "0"}):
            with self.assertRaisesRegex(
                AssertionError, "different library/environment pins"
            ):
                run(SimpleNamespace(resume=True, output=str(path)))


if __name__ == "__main__":
    unittest.main()
