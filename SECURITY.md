# Security Policy

## Supported Versions

This project is still evolving quickly. Security fixes are only guaranteed on the latest mainline pipeline code and recent schema generations.

| Version | Supported |
| ------- | --------- |
| v4.x    | Yes       |
| v3.x    | Limited   |
| < v3.0  | No        |

Notes:
- `v4.x` covers the current repo-grounded pipeline work and receives active fixes.
- `v3.x` may receive critical security fixes when practical, but no compatibility guarantees are made.
- Older versions should be upgraded rather than patched in place.

## Reporting a Vulnerability

Please do not open public GitHub issues for security vulnerabilities.

Report vulnerabilities privately to the project maintainer through one of these channels:
- GitHub Security Advisories / private vulnerability reporting, if enabled on the repository
- Direct maintainer email or other private contact channel listed in the repository profile

When reporting, include:
- A clear description of the issue
- Steps to reproduce
- Impact assessment
- Affected files, modules, or pipeline stage
- Proof-of-concept details if safe to share
- Any suggested remediation, if available

You can expect:
- Initial acknowledgement within 5 business days
- Follow-up once the issue is triaged
- A decision on severity and remediation approach after reproduction and impact review

Possible outcomes:
- Accepted: the issue is reproduced and scheduled for a fix
- Accepted with scope adjustment: the issue is valid but the remediation differs from the original report
- Declined: the report is not reproducible, is out of scope, or does not create a meaningful security impact

## Scope

Security reports are especially relevant for:
- Secret handling in `.env`, CI, and API client code
- Hugging Face upload credentials and dataset publishing flow
- GitHub issue ingestion and repository cloning logic
- Command execution, workspace isolation, and file write behavior
- Data leakage in traces, logs, or exported dataset rows

The following are generally out of scope unless they create a concrete exploit path:
- Model quality disagreements
- Labeling inconsistency without a security impact
- General dataset quality issues
- Rate-limit or quota exhaustion that does not expose data or privilege boundaries

## Handling Secrets

If you believe a secret has been exposed:
- Revoke and rotate the affected token immediately
- Remove it from local `.env` files, logs, CI settings, and any published artifacts
- Report the exposure privately with enough detail to identify where it may have been written

This repository should never publish live credentials in:
- Source-controlled files
- Dataset rows
- Run logs
- Example configuration snippets
