"""Compute a dataset of vulnerabilities of dependents of the security libraries."""

import json
import os
import re
import shutil
import subprocess
from collections import deque
from pathlib import Path
from typing import Literal

import git
from tqdm import tqdm
import pandas as pd

import cvar
import exceptions
import logger

LOG = logger.get()

DATA_DIR_ABSOLUTE = cvar.data_dir.resolve()


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

    def get_vulnerabilities(self,
                            security_library: Literal["PyCrypto", "PyNaCl", "M2Crypto", "cryptography"]
                            ) -> int | None:
        """
        Get the vulnerabilities of the repository related to given security library.
        Run licma in a Docker container, retrieve and parse the output, and remove the output file in the container.

        :param security_library: Target security library supported by licma.
        :return: Number of vulnerabilities found, or -1 on error.
        """
        LOG.debug("Running licma for repository %r and security library %r",
                  self.repo_name, security_library)

        security_library = security_library.lower()

        subprocess.run(["docker", "compose", "exec", "licma", "python3", "run_licma.py",
                        "--lc",  # Log to console.
                        "--la=py",  # Language.
                        "--ll=20",  # Debug level.
                        "-i", f"/usr/data/dependents/{self.repo_name}",
                        "--lib", security_library],
                       cwd=cvar.licma_dir,
                       check=True)

        licma_ls_output = subprocess.run(
            ["docker", "compose", "exec", "licma", "ls", "/usr/licma/output"],
            cwd=cvar.licma_dir,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8"
        )
        if licma_ls_output.returncode != 0:
            LOG.error(
                "Failed to list output files in licma container: %s",
                licma_ls_output.stderr
            )
            return None

        licma_output_file_name = licma_ls_output.stdout.strip()

        try:
            subprocess.run(
                ["docker", "cp", "licma:/usr/licma/output", str(DATA_DIR_ABSOLUTE / "licma")],
                check=True
            )
        except subprocess.CalledProcessError as exc:
            LOG.error(
                "Failed to copy output files from licma container: %s",
                exc
            )
            return None

        licma_output_path = cvar.data_dir / "licma" / "output" / licma_output_file_name
        if not licma_output_path.is_file():
            LOG.error(
                "Expected output file not found: %r",
                licma_output_path
            )
            return None

        output_file_path_repo = Path(str(licma_output_path.with_suffix('')) + "_" + self.repo_name).with_suffix(".csv")
        shutil.move(licma_output_path, output_file_path_repo)

        vulnerabilities = pd.read_csv(output_file_path_repo, encoding="utf-8", sep=";")

        subprocess.run(
            ["docker", "compose", "exec", "licma", "rm", "-rf", "/usr/licma/output"],
            cwd=cvar.licma_dir,
            check=True
        )

        return len(vulnerabilities)


def collect_vulnerabilities() -> None:
    """Collect vulnerabilities for dependents of security libraries."""
    security_libraries_path = cvar.data_dir / "licma_security_libraries.json"
    dependents_path = cvar.data_dir / "security_libraries_dependents.json"

    target_security_libraries_names = {"cryptography", "M2Crypto", "PyCrypto", "PyNaCl"}

    with open(security_libraries_path, "r", encoding="utf-8") as f:
        security_libraries: list[dict] = json.load(f)

    with open(dependents_path, "r", encoding="utf-8") as f:
        dependents: dict[str, dict] = json.load(f)

    security_libraries = [
        lib for lib in security_libraries if lib["name"] in target_security_libraries_names
    ]
    dependents = {
        lib_name: data
        for lib_name, data in dependents.items()
        if lib_name in target_security_libraries_names
    }

    vulnerabilities_out_path = cvar.data_dir / "vulnerabilities.json"

    subprocess.run(
        ["docker", "compose", "up", "-d"],
        cwd=cvar.licma_dir,
        check=True)

    collect_all_vulnerabilities(dependents, security_libraries, vulnerabilities_out_path)


def collect_all_vulnerabilities(dependents: dict[str, dict],
                                security_libraries: list[dict],
                                save_path: Path) -> None:
    """
    Collect vulnerabilities for all dependents of the target security libraries, and save.

    If save path exists, load existing progress and continue.
    """
    # Vulnerabilities, with keys as security library, and values as
    # dictionaries of form dependent: vulnerability count.
    all_vulnerabilities: dict[str, dict[str, int | None]] = {}

    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            all_vulnerabilities = json.load(f)

    for lib_name in dependents:
        all_vulnerabilities.setdefault(lib_name, {})

    pending_dependents: dict[str, deque[dict]] = {}
    for lib in security_libraries:
        lib_name = lib["name"]
        if lib_name not in dependents:
            raise ValueError(
                f"Dependents data not found for security library: {lib_name}"
            )
        processed_dependents = set(all_vulnerabilities.get(lib_name, {}))
        lib_queue: deque[dict] = deque()
        for dependent in dependents[lib_name]["dependents"]:
            dependent_name = dependent["name"].split("/")[-1]
            if dependent_name in processed_dependents:
                continue
            lib_queue.append(dependent)
        pending_dependents[lib_name] = lib_queue

    total_pending = sum(len(queue) for queue in pending_dependents.values())
    if total_pending == 0:
        LOG.info("All dependents already processed; nothing to do.")
        return

    progress = tqdm(total=total_pending, desc="Collecting vulnerabilities")

    while total_pending > 0:
        processed_in_cycle = False
        for lib in security_libraries:
            lib_name = lib["name"]
            queue = pending_dependents[lib_name]
            if not queue:
                continue

            dependent = queue.popleft()
            processed_in_cycle = True

            collector = VulnerabilitiesCollect(dependent["url"])
            dependent_vulnerabilities_count: int | None = collector.get_vulnerabilities(lib_name)
            dependent_name = dependent["name"].split("/")[-1]

            total_pending -= 1
            progress.update(1)

            all_vulnerabilities[lib_name][dependent_name] = dependent_vulnerabilities_count

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(all_vulnerabilities, f, indent=4)

            LOG.info("Collected vulnerabilities for dependent %r of library %r: counted %r",
                     dependent_name, lib_name, dependent_vulnerabilities_count)

        if not processed_in_cycle:
            break

    progress.close()

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(all_vulnerabilities, f, indent=4)


def main():
    collect_vulnerabilities()


if __name__ == "__main__":
    main()
