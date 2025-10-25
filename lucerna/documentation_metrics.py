"""Collect documentation metrics from a set of git repositories and save in data files."""

import json
import os
import re
from pathlib import Path
from typing import Iterable, Callable, TypedDict
import subprocess

import git
from interrogate import coverage
import pandas

import cvar
import logger

LOG = logger.get()


class DocumentationMetrics(TypedDict):
    """Type for documentation metrics dictionary."""

    repo_name: str
    readme_length: int
    readme_completeness: float
    docstring_coverage: float
    documentation_percentage: float
    has_docs: bool
    has_api_docs: bool


documentation_metrics_dictionary: DocumentationMetrics = {
    "repo_name": "",
    "readme_length": 0,
    "readme_completeness": 0.0,
    "docstring_coverage": 0.0,
    "documentation_percentage": 0.0,
    "has_docs": False,
    "has_api_docs": False,
}


class DocumentationMetricsCollect:
    """Collect documentation metrics from a git repository and save in a data file."""

    def __init__(self, repo_url: str):
        if not isinstance(repo_url, str) or not repo_url:
            raise ValueError(f"Received repository URL was empty or invalid: {repo_url!r}")

        self.repo_url = repo_url
        self.repo_name = Path(self.repo_url.rstrip("/")).stem

        self.repo_dir = cvar.data_dir / "repos" / "documentation_metrics" / self.repo_name
        self.repo_dir.parent.mkdir(parents=True, exist_ok=True)

        self.repo: git.Repo
        if os.path.isdir(self.repo_dir) and os.listdir(self.repo_dir):
            self.repo = git.Repo(self.repo_dir)
            LOG.debug("Opened existing repository %r in %r", self.repo_name, self.repo_dir)
        else:
            self.repo = git.Repo.clone_from(self.repo_url, self.repo_dir)
            LOG.debug("Cloned repository %r into %r", self.repo_url, self.repo_dir)

        main_branch = self.main_branch
        self.repo.git.checkout(main_branch)
        LOG.debug("Checked out main branch %r for repository %r", main_branch, self.repo_name)

        self.metrics = documentation_metrics_dictionary.copy()
        self.metrics["repo_name"] = self.repo_name

        with open(cvar.resources_dir / "readme_section_names.json", "r", encoding="utf-8") as f:
            self.readme_section_names = json.load(f)

        self.readme: str | None = None

    @property
    def main_branch(self) -> str:
        """Main, master, or default branch of the repository."""
        if not self.repo:
            raise ValueError("Repository must be initialized before retrieving the main branch.")

        candidates = ("main", "master")
        local_heads = {head.name for head in self.repo.heads}
        for candidate in candidates:
            if candidate in local_heads:
                return candidate

        # Default branch.
        show_result = self.repo.git.remote("show", "origin")
        matches = re.search(r"\s*HEAD branch:\s*(.*)", show_result)
        if matches:
            default_branch = matches.group(1).strip()
            if default_branch:
                return default_branch

        error_message = f"Could not find main branch for repository {self.repo_name}"
        LOG.error(error_message)
        raise ValueError(error_message)

    def collect_metrics(self) -> DocumentationMetrics:
        """Collect documentation metrics with per-metric fault isolation."""
        steps: Iterable[Callable[[], None]] = (
            self.get_readme_length,
            self.get_readme_completeness,
            self.get_docstring_coverage,
            self.get_documentation_percentage,
            # self.get_has_docs,
            # self.get_has_api_docs, # TODO implement
        )
        for step in steps:
            step()

        return self.metrics

    def _find_readme(self) -> str | None:
        """
        Find README file in the repository root directory.

        :return: Path to README file if found, else None.
        """
        if self.readme is not None:
            return self.readme

        candidates = {"readme", "read_me"}
        extensions = {"", ".md", ".rst", ".txt", ".markdown"}
        try:
            for entry in Path(self.repo_dir).iterdir():
                if not entry.is_file():
                    continue
                name = entry.name.lower()
                for base in candidates:
                    for ext in extensions:
                        if name == base + ext:
                            self.readme = str(entry)
                            return str(entry)
        except FileNotFoundError:
            LOG.error("Repository directory %r not found for repository %r when finding readme",
                      self.repo_dir, self.repo_name)
            raise

        LOG.warning("Could not find README file for repository %r in %r", self.repo_name, self.repo_dir)
        self.readme = ""
        return None

    def get_readme_length(self) -> None:
        readme_path = self._find_readme()

        if readme_path:
            with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            self.metrics["readme_length"] = len(content)
        else:
            self.metrics["readme_length"] = 0

    def get_readme_completeness(self) -> None:
        """Score fraction of common sections present [0.0, 1.0]."""
        readme_path = self._find_readme()
        if not readme_path:
            self.metrics["readme_completeness"] = 0.0
            return

        with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        content_lowercase = content.lower()
        # TODO it just checks if the section name appears anywhere, maybe try detecting only in sections?
        hits = sum(1 for s in self.readme_section_names if s in content_lowercase)
        self.metrics["readme_completeness"] = (
                hits / len(self.readme_section_names)) if self.readme_section_names else 0.0

    def get_docstring_coverage(self) -> None:
        """Calculate fraction of functions/methods with docstrings."""
        cov = coverage.InterrogateCoverage(paths=[str(self.repo_dir)])
        results = cov.get_coverage()
        self.metrics["docstring_coverage"] = results.perc_covered / 100.0

    def get_documentation_percentage(self) -> None:
        """Compute fraction of documentation lines across all Python files in our repo."""
        try:
            result = subprocess.run(
                [
                    "pygount",
                    "--suffix=py",
                    str(self.repo_dir),
                    "--format=json",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=180,
            )

            if result.returncode != 0:
                LOG.error(
                    "Pygount exited with code %s for %s: %s",
                    result.returncode,
                    self.repo_name,
                    (result.stderr or "").strip(),
                )
                raise Exception("Pygount execution failed.")

            data = json.loads(result.stdout or "{}")
            summary = data.get("summary") or {}

            total_code = summary.get("totalCodeCount")
            if not isinstance(total_code, int) or total_code == 0:
                LOG.error("Pygount found no source code for: %s", self.repo_name)
                raise Exception(f"Pygount found no source code for: {self.repo_name}")

            documentation_percentage = summary.get("totalDocumentationPercentage")

            if not isinstance(documentation_percentage, float):
                LOG.error("Invalid documentation percentage value: %s", documentation_percentage)
                raise

            self.metrics["documentation_percentage"] = documentation_percentage / 100.0

        except FileNotFoundError:
            LOG.error("Pygount executable not found.")
            raise
        except json.JSONDecodeError as exc:
            LOG.exception("Failed to parse pygount JSON for %s: %s", self.repo_name, exc)
            raise
        except Exception as exc:
            LOG.exception("Pygount analysis failed for %s: %s", self.repo_name, exc)
            raise


class DocumentationMetricsSchedule:
    """Schedule documentation metrics collection for a set of repositories."""

    def __init__(self, repo_urls: Iterable[str], output_csv: Path | None = None):
        self.repo_urls = list(repo_urls)

        self.output_csv = output_csv or (
                cvar.data_dir / "documentation_metrics.csv")
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        metrics_list: list[DocumentationMetrics] = []
        for url in self.repo_urls:
            try:
                collector = DocumentationMetricsCollect(url)
                repo_metrics = collector.collect_metrics()
                metrics_list.append(repo_metrics.copy())
            except Exception as exc:
                LOG.exception("Failed to collect metrics for %s: %s", url, exc)
                self.save_metrics(metrics_list)
                raise exc

        self.save_metrics(metrics_list)
        LOG.info("Wrote %d rows to %s", len(metrics_list), self.output_csv)

    def save_metrics(self, metrics_list: list[DocumentationMetrics]) -> None:
        df = pandas.DataFrame(metrics_list, columns=tuple(documentation_metrics_dictionary.keys()))
        df.to_csv(self.output_csv, index=False)


def _example_usage():
    repo_list = [
        "https://github.com/qwerltaz/metric-dynamics",
        "https://github.com/qwerltaz/lucerna"
    ]
    scheduler = DocumentationMetricsSchedule(repo_list)
    scheduler.run()


if __name__ == "__main__":
    _example_usage()
