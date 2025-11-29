"""Compute a dataset of vulnerabilities of dependents of the security libraries."""
import json
import os
import re
from pathlib import Path

import git
from tqdm import tqdm

import cvar
import exceptions
import logger

LOG = logger.get()


class VulnerabilitiesCollect:
    """Collect vulnerabilities from a GitHub repository."""

    def __init__(self, repo_url: str):
        if not isinstance(repo_url, str) or not repo_url:
            raise ValueError(
                f"Received repository URL was empty or invalid: {repo_url!r}"
            )

        self.repo_url = repo_url
        self.repo_name = Path(self.repo_url.rstrip("/")).stem

        self.repo_dir = (
                cvar.data_dir / "repos" / "dependents" / self.repo_name
        )
        self.repo_dir.parent.mkdir(parents=True, exist_ok=True)

        self.repo: git.Repo
        if os.path.isdir(self.repo_dir) and os.listdir(self.repo_dir):
            self.repo = git.Repo(self.repo_dir)
            LOG.debug(
                "Opened existing repository %r in %r", self.repo_name, self.repo_dir
            )
        else:
            try:
                self.repo = git.Repo.clone_from(self.repo_url, self.repo_dir)
                LOG.debug("Cloned repository %r into %r", self.repo_url, self.repo_dir)
            except git.exc.GitCommandError:
                LOG.error(
                    "Failed to clone: repository not found: %r into %r",
                    self.repo_url,
                    self.repo_dir,
                )
                raise

        python_files = list(self.repo_dir.rglob("*.py"))
        if not python_files:
            LOG.info(
                "No Python files found in repository %r at %r, skipping",
                self.repo_name,
                self.repo_dir,
            )
            raise exceptions.NoPythonFilesInRepository(
                f"No Python files found in repository {self.repo_name}"
            )

        main_branch = self.main_branch
        self.repo.git.checkout(main_branch)
        LOG.debug(
            "Checked out main branch %r for repository %r", main_branch, self.repo_name
        )

    @property
    def main_branch(self) -> str:
        """Default, main, or master branch of the repository."""
        if not self.repo:
            raise ValueError(
                "Repository must be initialized before retrieving the main branch."
            )

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

        error_message = (
            f"Could not find the default, main, or master branch for repository '{self.repo_name}' "
            f"from available branches {local_heads}."
        )
        raise ValueError(error_message)

    def get_vulnerabilities(self) -> list:
        """
        Get the vulnerabilities of the dependents of the repository.
        Return a dictionary of form dependent: vulnerability list.
        """
        raise NotImplementedError()


def collect_vulnerabilities() -> None:
    """Collect vulnerabilities for dependents of security libraries."""
    security_libraries_path = cvar.data_dir / "security_libraries_dependents_count.json"
    dependents_path = cvar.data_dir / "security_libraries_dependents.json"

    with open(security_libraries_path, "r", encoding="utf-8") as f:
        security_libraries: list[dict] = json.load(f)

    with open(dependents_path, "r", encoding="utf-8") as f:
        dependents_raw: dict[str, dict] = json.load(f)

    dependents = dict(sorted(dependents_raw.items(), key=lambda x: x[1]["dependents_count"]))
    security_libraries = sorted(security_libraries, key=lambda x: dependents[x["name"]]["dependents_count"])

    vulnerabilities_out_path = cvar.data_dir / "vulnerabilities.csv"

    # Vulnerabilities, with keys as security library, and values as
    # dictionaries of form dependent: list of vulnerabilities.
    all_vulnerabilities: dict[str, dict[str, list]] = {}

    for lib in tqdm(security_libraries):
        lib_name = lib["name"]
        if lib_name not in dependents:
            continue

        for dependent in dependents[lib_name]["dependents"]:
            dependent_repo_url = dependent["url"]

            collector = VulnerabilitiesCollect(dependent_repo_url)
            dependent_vulnerabilities = collector.get_vulnerabilities()
            if dependent_vulnerabilities:
                dependent_name = dependent["name"].split("/")[-1]
                all_vulnerabilities[lib_name][dependent[dependent_name]] = dependent_vulnerabilities


def main():
    collect_vulnerabilities()


if __name__ == "__main__":
    main()
