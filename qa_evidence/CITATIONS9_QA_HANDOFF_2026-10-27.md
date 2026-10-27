# OpenAI Knowledge Retrieval #9 — independent QA handoff

Date: 27 October 2026 (Türkiye)
Upstream PR: https://github.com/openai/openai-knowledge-retrieval/pull/9
Tested PR head: `d6936e6a21c265fc027456ac8ab13e158e4fea4c`
Compared against upstream base: `f62c5dd49955d2bc793e0a55989863dca61f1ead`
Independent QA branch: `qa/openai-citation-invariants-20261027` in `senanurcetin/openai-knowledge-retrieval`.

## Validation

A new test file, `qa_evidence/test_citation_invariants.py`, contains **152
parametrized / targeted cases**:

- 144 deterministic differential tests: 24 random shuffles x 6 limits
  (0, 1, 2, 3, 5, 99) over 35 interleaved chunks from 5 sources. An
  independent grouping oracle selects the first eligible matches for each
  source, restores original global citation order, and verifies complete
  citation dictionaries. Inputs are checked for mutation.
- 3 long-source starvation tests: 800 early matching chunks from one source
  cannot prevent a later source from receiving its quota.
- 3 nonpositive quota tests verify early exit even with malformed chunks.
- 1 test covers missing optional title/page and normal fallback to `None`.
- 1 test confirms that nonmatching chunks do not consume quota.

Combined with the 3 existing PR regression tests:

```powershell
$env:PYTHONPATH = 'qa_deps'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
py -3.13 -m pytest -q -o addopts= `
  tests/test_extract_spans.py `
  qa_evidence/test_citation_invariants.py `
  --tb=short --disable-warnings
```

**Observed result: 155 passed, 0 failed (0.36 s).**

Windows / Python 3.13, `rapidfuzz==3.14.3` installed into a temporary
`qa_deps` directory **inside the local QA workspace only**, not globally.
Ruff lint and format checks passed for the new file; `git diff --check`
also passed. Ruff emits a project-level deprecation notice about settings
in `pyproject.toml`, unrelated to this QA test.

## Negative control (A/B)

The three `test_very_long_matching_source_cannot_starve_late_sources`
variants were also run against unchanged upstream base commit
`f62c5dd` using the same test file and dependency installation:

**3 failed on the base, 3 passed on the PR head.** The baseline selects
only the first source's chunks before it considers the later source; the
PR does not have that global truncation behavior.

This is a focused negative control, not a claim that every new test was
run on the base commit or that the repository-wide suite was run this
turn. Earlier repository-wide results reported on #9 are historical.

## Independent scope / Notebook handoff

Notebook 1 can use these tests to check future changes to source quotas
or cherry-pick/adapt the strongest ones to normal `tests/` if an upstream
maintainer requests more cases. No edits were made to the live contributor
branch, no upstream PR or CI run was created, and no maintainer ping was
posted. Existing PR #9 remains the contribution awaiting upstream
maintainer review.

Additional low-priority observation: a matched chunk explicitly carrying
`metadata=None` causes `AttributeError` during title fallback, but that
code predates #9. The current Qdrant pipeline constructs a dictionary for
retrieved metadata. Do not attribute this unrelated edge case to #9 or
treat it as a verified production incident.
