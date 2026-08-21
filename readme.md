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


# Results and discussion

The final findings suggested that, to ensure security libraries are
used the most securely, their developers should:
1) Prioritize inline documentation: Focus on comprehensive docstrings and code comments (where necessary)
rather than relying solely on external documentation sites
or extensive READMEs.
2) Optimize for readability: Write documentation at an
accessible reading level. PyNaCl’s success with a Flesch-Kincaid readability score of 18.92 (the best among the studied security libraries) suggests that clearer, more readable documentation reduces misuse.
3) Document security-critical code: Ensure that cryptographic operations, security boundaries, and potential
pitfalls are documented at the point of use, not just in
overview documentation.

The analysis reveals an interesting pattern linking documentation characteristics to security outcomes. PyNaCl, which demonstrated the lowest vulnerability rate, combines three critical documentation strengths: high readability (Flesch-Kincaid score of 18.92) and substantial documentation percentage (0.21, on the higher end of the distribution). This suggests that making documentation accessible and embedding it within the codebase itself are key factors in preventing misuse. Conversely, cryptography, despite being the most actively maintained library with the highest documentation up-to-date score (0.77), suffered the most vulnerabilities. This library had the lowest docstring coverage (0.06) and documentation percentage (0.03). This indicates a potential over-reliance on external documentation at the expense of inline guidance where developers need it most. This result undermines the assumption that simply keeping documentation up to date is sufficient. The location and accessibility of documentation might be equally important. Notably, README characteristics (length and completeness) showed no clear correlation with security outcomes. PyNaCl's minimal README (889 characters) corresponded with the lowest vulnerability rate, while PyCrypto's extensive README (4949 characters) did not prevent substantial misuse. This indicates that high-level overviews may be less important than documentation embedded within the code, especially given that in-source documentation is also what is readily and instantly available when creating software and using the library. The evidence supports the claim that documentation quality affects security library misuse, with key factors being revealed: (1) readability of external documentation, and (2) presence of docstrings and inline comments (where necessary). Libraries should prioritize these characteristics over just external documentation or extensive READMEs. This aligns with the anecdotal consideration that, for some developers, the process of incorporating a library might be as follows: find out about a library, download and import it, read the README, and, when using it, just look at the docstrings for explanation, without considering reading the external documentation.

<img width="50%" height="50%" alt="image" src="https://github.com/user-attachments/assets/acfb99e4-76b8-4c00-8aa4-b32abe8ebfd9" />

<img width="516" height="222" alt="image" src="https://github.com/user-attachments/assets/db66842d-eb8f-4225-935f-24c093d8066f" />

<img width="522" height="292" alt="image" src="https://github.com/user-attachments/assets/40ffc0f0-9b1e-492b-8aa4-08f238b2af9a" />

The ’computed’ row excludes dependents for which
vulnerabilities could not be computed. The ’not computed’
row shows the number of dependents for which vulnerabilities
could not be computed. The non-zero row shows the number
of dependents with at least one vulnerability. The non-zero
percentage row shows the percentage of computed dependents
that have at least one vulnerability. The ’average’ row shows
the average number of vulnerabilities across the dependents of
that security library.

<img width="1466" height="1766" alt="image" src="https://github.com/user-attachments/assets/5b5fa936-caa8-4611-b9e0-d02d39a2e2f3" />
Distributions of the documentation metrics for all security libraries.
