# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Please report vulnerabilities by emailing **srivnamrata@yahoo.co.in** with the subject line `[multixtract] Security vulnerability`. Include:

- A description of the vulnerability and its potential impact
- Steps to reproduce (minimal example preferred)
- Any suggested mitigation or fix

You will receive an acknowledgement within 72 hours. We aim to release a patch within 14 days for confirmed critical issues.

## Scope

This policy covers the `multixtract` Python package published on PyPI. It does not cover third-party dependencies (PyMuPDF, python-docx, OpenAI SDK, etc.) — please report those to their respective maintainers.

## Security considerations for users

- **Local vision providers:** All local providers (`SmolVLMVisionModel`, `Llama32VisionModel`, `Qwen2VLVisionModel`) load model weights from HuggingFace. Pin `model_id` to a specific revision in production environments to avoid unexpected upstream changes.
- **Credentials:** Never hard-code API keys. Pass them via environment variables or a secrets manager. The Azure providers support `azure_ad_token_provider` for managed-identity / passwordless auth.
- **File inputs:** multixtract processes user-supplied files (PDF, DOCX, PPTX, XLSX). Treat untrusted files with the same caution as any user-supplied content.
