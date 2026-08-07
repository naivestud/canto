# Security Policy

## Supported versions

canto is pre-1.0. Only the latest release receives security fixes.

| Version | Supported |
| --- | --- |
| latest  | yes |
| older   | no |

## Reporting a vulnerability

Please do **not** open a public issue for a security problem.

Email the maintainers using the contact details on their GitHub profile, or
use GitHub's private vulnerability reporting feature on the repository
(**Security → Report a vulnerability**). You should receive an acknowledgement
within a few days, and we will work with you on a fix and a coordinated
disclosure timeline.

## Scope notes

- Loading a checkpoint executes `torch.load`; only load checkpoints you trust.
  canto passes `weights_only=True`, which blocks arbitrary code execution but
  you should still treat untrusted checkpoints as suspicious.
- canto reads and writes WAV files from paths you provide; run it with the
  filesystem permissions you are comfortable with.
