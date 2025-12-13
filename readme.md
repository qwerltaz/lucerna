# About

Are badly documented Python security libraries used less securely? This project uses a static code analysis tool, LICMA,
to analyze several Python security libraries, focusing on their documentation and usage patterns in open-source
projects. The goal is to identify potential correlations between the quality of documentation and the security of code
that uses these libraries.

# Requirements

- docker
- docker-compose
- Python 3.14
- uv

# Installation

1. Clone this repository.
2. Set up LICMA, located in `./licma`.
   See https://github.com/stg-tud/licma/blob/b899e6e682f7716d19e79d6ce7b73c28c6efd4cf/README.md.

# Usage

1. Run one of the scripts in `./lucerna` to create a corresponding dataset with `cd lucerna` and
   `uv run <script_name>.py`. See inside the scripts for more information inside docstrings.