"""Collect documentation metrics from a set of git repositories and save in data files."""

import json
import os
import re
from pathlib import Path
from typing import Iterable, Callable, TypedDict
import subprocess

import git
from git import GitCommandError
from interrogate import coverage
import pandas

import cvar
import debug_tools
import logger

LOG = logger.get()


class DocumentationMetrics(TypedDict):
    """Type for documentation metrics dictionary."""

    repo_name: str
    readme_length: int
    github_wiki_length: int
    readme_completeness: float
    docstring_coverage: float
    documentation_percentage: float


documentation_metrics_dictionary: DocumentationMetrics = {
    "repo_name": "",
    "readme_length": 0,
    "github_wiki_length": 0,
    "readme_completeness": 0.0,
    "docstring_coverage": 0.0,
    "documentation_percentage": 0.0,
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

        self.readme_path: str | None = self._find_readme()

        self.wiki_dir: Path = self.repo_dir.parent / f"{self.repo_name}.wiki"

    @property
    def main_branch(self) -> str:
        """Default, main, or master branch of the repository."""
        if not self.repo:
            raise ValueError("Repository must be initialized before retrieving the main branch.")

        # Default branch.
        origin_info = self.repo.git.remote("show", "origin")
        matches = re.search(r"\s*HEAD branch:\s*(.*)", origin_info)
        if matches:
            default_branch = matches.group(1).strip()
            if default_branch:
                return default_branch

        candidates = ("main", "master")
        local_heads = {head.name for head in self.repo.heads}
        for candidate in candidates:
            if candidate in local_heads:
                return candidate

        error_message = (f"Could not find the default, main, or master branch for repository '{self.repo_name}' "
                         f"from available branches {local_heads}.")
        raise ValueError(error_message)

    def collect_metrics(self) -> DocumentationMetrics:
        """Collect documentation metrics with per-metric fault isolation."""
        steps: Iterable[Callable[[], None]] = (
            self.get_readme_length,
            self.get_github_wiki_length,
            self.get_readme_completeness,
            self.get_docstring_coverage,
            self.get_documentation_percentage,
        )
        for step in steps:
            step()

        return self.metrics

    def _find_readme(self) -> str | None:
        """
        Find README file in the repository root directory.

        :return: Path to README file if found, else None.
        """
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
                            return str(entry)
        except FileNotFoundError:
            LOG.error("Repository directory %r not found for repository %r when finding readme",
                      self.repo_dir, self.repo_name)
            debug_tools.debug_raise()

        LOG.warning("Could not find README file for repository %r in %r", self.repo_name, self.repo_dir)
        return None

    def get_readme_length(self) -> None:
        """Calculate length of README file in characters."""
        readme_path = self.readme_path

        if readme_path:
            with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            self.metrics["readme_length"] = len(content)
        else:
            self.metrics["readme_length"] = 0

    def get_github_wiki_length(self) -> None:
        """Calculate total character length of the GitHub wiki pages; 0 if no wiki."""
        wiki_url = self._github_wiki_url()
        if not wiki_url:
            self.metrics["github_wiki_length"] = 0
            return

        try:
            if os.path.isdir(self.wiki_dir) and os.listdir(self.wiki_dir):
                wiki_repo = git.Repo(self.wiki_dir)
                try:
                    wiki_repo.remotes.origin.fetch()
                except Exception as exc:
                    LOG.warning("Fetch of wiki for `%s`: %s", self.repo_name, exc)
            else:
                git.Repo.clone_from(wiki_url, self.wiki_dir)
        except GitCommandError:
            LOG.info("Wiki does not exist for %s", self.repo_name)
            self.metrics["github_wiki_length"] = 0
            return
        except Exception as exc:
            LOG.warning("Wiki not found for `%s`: %s", self.repo_name, exc)
            self.metrics["github_wiki_length"] = 0
            return

        wiki_files_text_extensions = {
            ".md",
            ".markdown",
            ".mdown",
            ".mkdn",
            ".rst",
            ".txt",
            ".adoc",
            ".asciidoc",
            ".adoc",
            ".asc",
            ".mediawiki",
            ".wiki",
            ".textile",
            ".creole",
            ".org",
            ".pod",
        }
        total_characters = 0
        try:
            for root, dirs, files in os.walk(self.wiki_dir):
                if ".git" in dirs:
                    dirs.remove(".git")

                for file_name in files:
                    extension = Path(file_name).suffix.lower()
                    if extension not in wiki_files_text_extensions:
                        continue
                    file_path = Path(root) / file_name
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                            total_characters += len(f.read())
                    except Exception as exc:
                        LOG.debug("Skipping unreadable wiki file %s: %s", file_path, exc)
        except Exception as exc:
            LOG.debug("Failed while scanning wiki for %s: %s", self.repo_name, exc)

        self.metrics["github_wiki_length"] = total_characters

    def _github_wiki_url(self) -> str | None:
        """Build the wiki git URL if the repository is hosted on GitHub; else None."""
        url = self.repo_url.rstrip("/")

        base = url[:-4] if url.endswith(".git") else url

        # HTTPS form: https://github.com/owner/repo
        https_match = re.match(r"^https://github\.com/([^/]+)/([^/]+)$", base, re.IGNORECASE)
        if https_match:
            owner, repo = https_match.groups()
            return f"https://github.com/{owner}/{repo}.wiki.git"

        return None

    def get_readme_completeness(self) -> None:
        """Score fraction of common sections present [0.0, 1.0]."""
        readme_path = self.readme_path
        if not readme_path:
            self.metrics["readme_completeness"] = 0.0
            return

        with open(readme_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # TODO it just checks if the section name appears anywhere, maybe try detecting only in sections?
        hits = sum(1 for s in self.readme_section_names if re.search(s, content, re.RegexFlag.IGNORECASE))

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
                debug_tools.debug_raise()

            self.metrics["documentation_percentage"] = documentation_percentage / 100.0

        except FileNotFoundError:
            LOG.error("Pygount executable not found.")
            debug_tools.debug_raise()
        except json.JSONDecodeError as exc:
            LOG.exception("Failed to parse pygount JSON for %s: %s", self.repo_name, exc)
            debug_tools.debug_raise()
        except Exception as exc:
            LOG.exception("Pygount analysis failed for %s: %s", self.repo_name, exc)
            debug_tools.debug_raise()


def schedule_documentation_metrics(repo_urls: Iterable[str], output_csv: Path | None = None) -> None:
    """Schedule documentation metrics collection for a set of repositories and write to CSV."""
    output_csv = output_csv or (cvar.data_dir / "documentation_metrics.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    metrics_list: list[DocumentationMetrics] = []
    for url in repo_urls:
        try:
            collector = DocumentationMetricsCollect(url)
            repo_metrics = collector.collect_metrics()
            metrics_list.append(repo_metrics.copy())
        except Exception as exc:
            LOG.exception("Failed to collect metrics for %s: %s", url, exc)
            _save_metrics(metrics_list, output_csv)
            debug_tools.debug_raise()

    _save_metrics(metrics_list, output_csv)


def _save_metrics(metrics_list: list[DocumentationMetrics], output_csv: Path | None):
    df = pandas.DataFrame(metrics_list, columns=tuple(documentation_metrics_dictionary.keys()))
    df.to_csv(output_csv, index=False)
    LOG.info("Saved documentation metrics. Wrote %d rows to %s", len(metrics_list), output_csv)


def _example_usage():
    repo_list = [
        "https://github.com/certifi/python-certifi/",
        "https://github.com/qwerltaz/metric-dynamics",
        "https://github.com/qwerltaz/lucerna"
    ]
    schedule_documentation_metrics(repo_list)


if __name__ == "__main__":
    _example_usage()
