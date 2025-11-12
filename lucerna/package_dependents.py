"""Create the dataset of the security libraries and all their dependents."""

import json
import subprocess
from typing import TypedDict
from urllib.parse import urlparse

from process_security_libraries_dataset import load_security_libraries_df
import cvar
import logger

LOG = logger.get()


class LibraryDependent(TypedDict):
    """A dependent library with its name of form <owner/repo> and URL."""

    name: str
    url: str


class LibraryDependentsData(TypedDict):
    """Data and dependents of a library."""

    dependents_count: int
    dependents: list[LibraryDependent]


def _extract_owner_repo_from_url(github_url: str) -> str | None:
    """Return "owner/repo" from a GitHub repo URL, or None if invalid."""
    if not isinstance(github_url, str) or not github_url:
        return None

    parsed = urlparse(github_url.strip())
    if parsed.netloc.lower() != "github.com":
        return None

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        return None

    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]

    return f"{owner}/{repo}"


def _dep_item_to_name_url(item: dict) -> LibraryDependent:
    """Normalize a dependent item from github-dependents-info to {name, url}."""
    owner = item.get("owner")
    repo_name = item.get("repo_name")
    name = item.get("name")

    if owner and repo_name:
        dep_name = f"{owner}/{repo_name}"
        dep_url = f"https://github.com/{dep_name}"
        return {"name": dep_name, "url": dep_url}

    if isinstance(name, str) and "/" in name:
        owner2, repo2 = name.split("/", 1)
        dep_url = f"https://github.com/{owner2}/{repo2}"
        return {"name": name, "url": dep_url}

    if name is None:
        name = ""

    return {"name": name, "url": ""}


def get_library_dependents(github_url: str) -> list[LibraryDependent]:
    """
    Get all public dependents for a given library from its GitHub URL using the
    `github-dependents-info` CLI.

    :param github_url: URL of the library's GitHub repository.
    :return: List of dependents; for each, keys: "name" (owner/repo) and "url".
    """
    repo_slug = _extract_owner_repo_from_url(github_url)
    if not repo_slug:
        raise ValueError(f"Invalid GitHub URL: {github_url}")

    cmd = [
        "github-dependents-info",
        "--repo",
        repo_slug,
        "--json",
    ]

    try:
        LOG.debug("Running: %s", " ".join(cmd))
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "github-dependents-info CLI not found in PATH. Please install it to continue."
        ) from exc
    except Exception as exc:
        LOG.exception(
            "Failed executing github-dependents-info for %s: %s", repo_slug, exc
        )
        raise

    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(
            f"github-dependents-info failed for {repo_slug} with code {proc.returncode}: {stderr}"
        )

    raw = proc.stdout.strip()
    if not raw:
        LOG.info("No output received from github-dependents-info for %s", repo_slug)
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        LOG.error("Invalid JSON output from github-dependents-info for %s", repo_slug)
        return []

    items = data.get("all_public_dependent_repos") or []
    if not isinstance(items, list):
        LOG.error(
            "Unexpected schema: all_public_dependent_repos is not a list for %s",
            repo_slug,
        )
        return []

    result = [_dep_item_to_name_url(it) for it in items]
    LOG.info("Found %d dependents for %s", len(result), repo_slug)
    return result


def main():
    """Build mapping of security libraries to their list of dependents and save to json."""
    file_name = "security_libraries_dependents_count_tiny"
    file_path = (cvar.data_dir / file_name).with_suffix(".json")

    libraries_dependents_count = load_security_libraries_df(file_path)

    libraries_dependents: dict[str, LibraryDependentsData] = {}

    for _, row in libraries_dependents_count.iterrows():
        lib_name = row.get("name")
        repo_url = row.get("repo_url")

        if not isinstance(repo_url, str) or not repo_url:
            LOG.debug("Skipping library without repo_url: %s", lib_name)
            continue

        if not isinstance(lib_name, str) or not lib_name:
            slug = _extract_owner_repo_from_url(repo_url) or repo_url
            lib_name = slug

        dependents = get_library_dependents(repo_url)
        dependents_and_info = {
            "dependents_count": len(dependents),
            "dependents": dependents,
        }
        libraries_dependents[lib_name] = dependents_and_info

    output_path = (cvar.data_dir / "security_libraries_dependents").with_suffix(".json")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(libraries_dependents, f, indent=4, ensure_ascii=False)
        LOG.info("Saved libraries dependents to %s", output_path)
    except Exception:
        LOG.exception("Failed to write output JSON to %s", output_path)
        print(libraries_dependents) # Print everything fallback why not.


if __name__ == "__main__":
    main()
