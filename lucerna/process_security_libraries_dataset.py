"""
Process the raw security libraries dataset from Google's BigQuery.

Extract GitHub repository URLs from the dataset.

Save the processed data.
"""

import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

import logger

LOG = logger.get()

PROJECT_URLS_PREFIX_CANDIDATES = [  # From most to least likely to be the repo url.
    "homepage",  # Often, the repo is linked here.
    "source",
    "source code",
    "github",
    "repository",
    "code",
    "home",
    "documentation",
]


def _is_valid_repo_url(url: str) -> bool:
    repo_regex = r"^https://github\.com/([^/]+)/([^/]+)$"
    ret = bool(re.match(repo_regex, url))
    if not ret:
        LOG.debug("URL does not match git repo pattern: %s", url)

    return ret


def _process_repo_url(url: str | None) -> str | None:
    """Normalize various GitHub URLs to https://github.com/{owner}/{repo}"""
    if not isinstance(url, str) or not url:
        return None

    raw = url.strip()
    parsed = urlparse(raw)

    if parsed.netloc.lower() != "github.com":
        LOG.debug("URL is not a GitHub URL: %s", raw)
        return None

    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        LOG.debug("GitHub URL does not contain owner and repo: %s", raw)
        return None

    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    normalized = f"https://github.com/{owner}/{repo}"

    return normalized if _is_valid_repo_url(normalized) else None


def _extract_repo_url(row: pd.Series) -> str | None:
    project_urls = row.get("project_urls")

    assert isinstance(project_urls, list)

    for item in project_urls:
        if "," not in item:
            continue
        key, value = item.split(",", 1)
        key, value = map(str.strip, (key, value))
        for prefix in PROJECT_URLS_PREFIX_CANDIDATES:
            if key.lower().startswith(prefix):
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


def main():
    """Process the raw security libraries dataset."""
    security_libraries_raw_path = Path("../data/security_libraries_raw.json")
    security_libraries_processed_path = Path("../data/security_libraries.json")

    with open(security_libraries_raw_path, "r", encoding="utf-8") as f:
        security_libraries_raw_json = json.load(f)

    security_libraries = pd.DataFrame(security_libraries_raw_json)

    security_libraries["repo_url"] = security_libraries.apply(_extract_repo_url, axis=1)

    security_libraries = security_libraries.drop(
        columns=["download_url", "project_urls"], errors="ignore"
    )

    security_libraries = security_libraries.dropna(subset=["repo_url"])

    # No pandas.to_json as it adds unwanted escape characters before slashes.
    with open(security_libraries_processed_path, "w", encoding="utf-8") as f:
        json.dump(
            security_libraries.to_dict(orient="records"),
            f,
            indent=4,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
