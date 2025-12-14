"""Correlation of security library documentation metrics and dependents vulnerabilities."""

import json

import pandas as pd

import cvar

doc_metrics = pd.read_csv(cvar.data_dir / "licma_documentation_metrics.csv")
with open(cvar.data_dir / "vulnerabilities.json", "r", encoding="utf-8") as fh:
    vulnerabilities = json.load(fh)

vulnerabilities.pop("M2Crypto")

avg_misuses = (
    pd.Series({
        lib.lower(): pd.Series(vals.values(), dtype="float").dropna().mean()
        for lib, vals in vulnerabilities.items()
    })
    .rename("avg_misuses")
)

df = (
    doc_metrics.assign(repo_name=lambda d: d["repo_name"].str.lower())
    .merge(avg_misuses, left_on="repo_name", right_index=True)
)

doc_cols = [
    "readme_length", "readme_completeness",
    "docstring_coverage", "documentation_percentage",
    "documentation_up_to_date", "documentation_readability",
    # "github_wiki_length", "code_examples_ratio"  # These two have only 0 as values.
]

pearson_corr = df[doc_cols].corrwith(df["avg_misuses"], method="spearman")
print(pearson_corr)
