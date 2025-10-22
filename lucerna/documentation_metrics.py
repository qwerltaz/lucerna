"""Collect documentation metrics from a set of git repositories and save in data files."""

import os
import re

import git

import cvar
import logger

LOG = logger.get()


class DocumentationMetricsCollect:
    """Collect documentation metrics from a git repository and save in a data file."""

    def __init__(self, repo_url: str):
        if not isinstance(repo_url, str) or not repo_url:
            raise ValueError(f"Received repository URL was empty or invalid: {repo_url!r}")

        self.repo_url = repo_url
        self.repo_name = self.repo_url.strip("/").split("/")[-1]
        self.repo_dir = cvar.data_dir / "repos" / "documentation_metrics" / self.repo_name

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

    @property
    def main_branch(self) -> str:
        """Main, master, or default branch of the repository."""
        if not self.repo:
            raise ValueError("Repository must be initialized before retrieving the main branch.")

        candidates = ["main", "master", "origin/main", "origin/master"]
        refs = self.repo.references
        for candidate in candidates:
            if candidate in refs:
                return candidate

        # Default branch.
        show_result = self.repo.git.remote("show", "origin")
        matches = re.search(r"\s*HEAD branch:\s*(.*)", show_result)
        if matches:
            default_branch = matches.group(1)
            if default_branch:
                return default_branch

        error_message = f"Could not find main branch for repository {self.repo_name}"
        LOG.error(error_message)
        raise ValueError(error_message)

    def collect_metrics(self):
        """Collect documentation metrics and save to a data file."""


class DocumentationMetricsSchedule:
    """Schedule documentation metrics collection for a set of repositories."""

    def __init__(self):
        raise NotImplementedError()
