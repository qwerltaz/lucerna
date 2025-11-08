"""
Process the raw security libraries dataset from Google's BigQuery.

Extract GitHub repository URLs from the dataset.

Save the processed data.
"""

import json
from pathlib import Path
from typing import Dict
from urllib.parse import urlparse
import urllib

import pandas as pd

import logger
import cvar

LOG = logger.get()

DEPENDENTS_RESPONSE_EXPECTED_KEYS = [
    "dependentCount",
    "directDependentCount",
    "indirectDependentCount",
]


def _process_repo_url(url: str | None) -> str | None:
    """Normalize various GitHub URLs to https://github.com/{owner}/{repo}"""
    if not isinstance(url, str) or not url:
        return None

    url_stripped = url.strip()
    parsed = urlparse(url_stripped)

    if parsed.netloc.lower() != "github.com":
        LOG.debug("URL is not a GitHub URL: %s", url_stripped)
        return None

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        LOG.debug("GitHub URL does not contain owner and repo: %s", url_stripped)
        return None

    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    normalized = f"https://github.com/{owner}/{repo}"

    return normalized


def _extract_repo_url(row: pd.Series) -> str | None:
    project_urls = row.get("project_urls")

    assert isinstance(project_urls, list)

    for item in project_urls:
        if "," not in item:
            continue
        key, value = item.split(",", 1)
        key, value = map(str.strip, (key, value))
        processed = _process_repo_url(value)
        if processed:
            return processed

    download_url = row.get("download_url")
    processed_download = _process_repo_url(download_url)
    if processed_download:
        return processed_download

    LOG.debug(
        "Could not find any repo url for download_url: %s, project_urls: %s",
        str(row["download_url"]),
        str(row["project_urls"]),
    )
    return None


def process_raw(file_name: str, processed_file_name: str):
    """Process the raw security libraries dataset."""
    file_path = (cvar.data_dir / file_name).with_suffix(".json")
    processed_path = (cvar.data_dir / processed_file_name).with_suffix(".json")

    security_libraries = load_security_libraries_df(file_path)

    security_libraries["repo_url"] = security_libraries.apply(_extract_repo_url, axis=1)

    security_libraries = security_libraries.drop(
        columns=["download_url", "project_urls"], errors="ignore"
    )

    security_libraries = security_libraries.dropna(subset=["repo_url"])
    security_libraries = security_libraries.drop_duplicates(subset=["repo_url"])

    # Turn Nan into JSON-friendly None.
    security_libraries = security_libraries.where(pd.notnull(security_libraries), None)

    # No pandas.to_json, as it adds unwanted escape characters before slashes.
    with open(processed_path, "w", encoding="utf-8") as f:
        json.dump(
            security_libraries.to_dict(orient="records"),
            f,
            indent=4,
            ensure_ascii=False,
        )

    LOG.info("Processed raw dataset and saved to %s", processed_path)


def _fetch_dependents_counts(
    name: str | None,
    version: str | None,
) -> Dict[str, int] | None:
    """Call deps.dev API to fetch dependent counts for a PyPI package version.

    :param name: The name of the PyPI package.
    :param version: The version of the PyPI package.
    """
    assert name is not None, "Package name must be provided"
    assert version is not None, "Package version must be provided"

    base_url = "https://api.deps.dev/v3alpha/systems/pypi/packages"
    url = f"{base_url}/{name}/versions/{version}:dependents"

    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            if resp.status != 200:
                raise urllib.error.HTTPError(
                    url, resp.status, "Bad status", resp.headers, None
                )
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError:
        LOG.info("Skipping package dependents as it no longer exists: %s", name)
        return None

    assert list(data.keys()) == DEPENDENTS_RESPONSE_EXPECTED_KEYS, (
        f"Unexpected keys in dependents response: {data}"
    )

    LOG.debug("Fetched dependents counts for %s==%s: %s", name, version, data)
    return data


def add_dependents_col(file_name: str, processed_file_name: str):
    """Augment processed dataset with deps.dev dependent count columns and write JSON."""
    file_path = (cvar.data_dir / file_name).with_suffix(".json")
    processed_file_path = (cvar.data_dir / processed_file_name).with_suffix(".json")

    security_libraries = load_security_libraries_df(file_path)

    counts_df = security_libraries.apply(
        lambda r: _fetch_dependents_counts(r.get("name"), r.get("version")),
        axis=1,
        result_type="expand",
    )

    security_libraries[DEPENDENTS_RESPONSE_EXPECTED_KEYS] = counts_df[
        DEPENDENTS_RESPONSE_EXPECTED_KEYS
    ]

    security_libraries = security_libraries.dropna(
        subset=["dependentCount", "directDependentCount", "indirectDependentCount"]
    )

    # Turn Nan into JSON-friendly None.
    security_libraries = security_libraries.where(pd.notnull(security_libraries), None)

    with open(processed_file_path, "w", encoding="utf-8") as f:
        json.dump(
            security_libraries.to_dict(orient="records"),
            f,
            indent=4,
            ensure_ascii=False,
        )

    LOG.info(
        "Augmented dataset with dependents counts and saved to %s", processed_file_path
    )


def load_security_libraries_df(file_path: Path | str) -> pd.DataFrame:
    # upload_time dtype defaults to timestamp, which is not JSON serializable.
    security_libraries: pd.DataFrame = pd.read_json(
        file_path, dtype={"upload_time": str, "metadata_version": str}
    )
    return security_libraries


def main():
    """End-to-end: process raw then augment with dependents counts."""
    file_name = "security_libraries"
    raw_file_name = file_name + "_raw"
    file_name_with_dependents_count = file_name + "_dependents_count"

    process_raw(raw_file_name, file_name)
    add_dependents_col(file_name, file_name_with_dependents_count)


if __name__ == "__main__":
    main()
