---
title: Contributing to HEIR
weight: 20
---

There are several ways to contribute to HEIR, including:

- Discussing high-level designs and questions on HEIR's
  [discussions page](https://github.com/google/heir/discussions)
- Improving or expanding HEIR's documentation
- Contributing to HEIR's [code-base](https://github.com/google/heir)
- Discuss project direction at HEIR's
  [Working Group meetings](https://heir.dev/community/)

## Ways to contribute

We welcome pull requests, and have tagged issues for newcomers:

- [Good first issue](https://github.com/google/heir/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
- [Contributions welcome](https://github.com/google/heir/issues?q=is%3Aissue+is%3Aopen+label%3A%22contributions+welcome%22)
- [Research synthesis](https://github.com/google/heir/labels/research%20synthesis):
  determine what parts of recent FHE research papers can or should be ported to
  HEIR.

For new proposals, please open a GitHub
[issue](https://github.com/google/heir/issues) or start a
[discussion](https://github.com/google/heir/discussions) for feedback.

## Contributing code to HEIR

The following steps should look familiar to typical workflows for pull request
contributions. Feel free to consult
[GitHub Help](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests)
if you need more information using pull requests. HEIR-specific processes begin
at the [pull request review stage](#pull-request-review-flow).

### Setup

1. Fork the HEIR repository by clicking the **Fork** button on the
   [repository page](https://github.com/google/heir). This creates a copy of the
   HEIR repository on your own GitHub account, where you can make changes.

   <details>
   <summary>Setting up git to work with fork and upstream remotes.</summary>
    If you have cloned your fork, you will want to
     add the HEIR repository as an upstream remote:

   ```bash
   git remote add upstream https://www.github.com/google/heir
   ```

   Alternatively, if you have cloned the main HEIR repo, you can add your fork
   as a remote like this:

   ```bash
   git remote rename origin upstream
   git remote add origin https://www.github.com/<USERNAME>/heir
   ```

   Either way, you will want to create a development branch for your change:

   ```bash
   git checkout -b name-of-change
   ```

   In the remainder of this document, we will assume `origin` is your fork, and
   `upstream` is the main HEIR repo.

   </details>

1. See [Development](https://heir.dev/docs/development/) for information on
   installing developer dependencies, building and running tests, and adding new
   dialects or passes.

1. Sign the
   [Contributor License Agreement](https://cla.developers.google.com/about)
   (CLA). If you are working on HEIR as part of your employment, you might have
   to instead sign a Corporate CLA. See more
   [here](https://github.com/google/heir/blob/main/CONTRIBUTING.md#sign-our-contributor-license-agreement).

### Preparing a pull request

1. Sync your changes against the upstream HEIR repository, i.e., make sure your
   contributions are (re)based of the most recent `upstream/main` commit.

1. Check HEIR's lint and style checks by running the following from the top of
   the repository:

   ```bash
    pre-commit run --all
   ```

1. Make sure tests are passing with the following:

   ```bash
    bazel test @heir//...
   ```

1. Once you are ready with your change, create a commit, e.g.:

   ```bash
   git add change.cpp
   git commit -m "Detailed commit message"
   git push --set-upstream origin name-of-change
   ```

### Pull request review flow

1. **New PR**:

- When a new PR is submitted, it is inspected for quality requirements, such as
  the CLA requirement, and a sufficient PR description.
- If the PR passes checks, we assign a reviewer. If not, we request additional
  changes to ensure the PR passes CI checks.

2. **Review**

- A reviewer will check the PR and potentially request additional changes.
- If a change is needed, the contributor is requested to make a suggested
  change. Please make changes with additional commits to your PR, to ensure that
  the reviewer can easily see the diff.
- If all looks good, the reviewer will approve the PR.
- This cycle repeats itself until the PR is approved.

3. **Approved**

- **At this stage, you must squash your commits into a single commit.**
- Once the PR is approved, a GitHub workflow will
  [check](https://github.com/google/heir/blob/main/.github/workflows/pr_review.yml)
  your PR for multiple commits. You may use the `git rebase -i` to squash the
  commits. Pull requests must consist of a single git commit before merging.

4. **Pull Ready**

- Once the PR is squashed into a single git commit, a maintainer will apply the
  `pull ready` label.
- This initiates the internal code migration and presubmits.
- After the internal process is finished, the commit will be added to `main` and
  the PR closed as merged by that commit.

### Internal review details

This diagram summarizes the GitHub/Google code synchronization process. This is
largely automated by a Google-owned system called
[Copybara](https://github.com/google/copybara), the configuration for which is
Google-internal. This system treats the Google-internal version of HEIR as the
source of truth, and applies specified transformation rules to copy internal
changes to GitHub and integrate external PRs internally.

Notable aspects:

- The final merged code may differ slightly from a PR. The changes are mainly to
  support stricter internal requirements for BUILD files that we cannot
  reproduce externally due to minor differences between Google's internal build
  systems and bazel that we don't know how to align. Sometimes they will also
  include additional code quality fixes suggested by internal static analyzers
  that do not exist outside of Google.
- Due to the above, signed commits with internal modifications will not maintain
  valid signatures after merging, which labels the commit with a warning.
- You will see various actions taken on GitHub that include `copybara` in the
  name, such as changes that originate from Google engineers doing various
  approved migrations (e.g., migrating HEIR to support changes in MLIR or
  abseil).

{{% figure src="/images/heir-copybara.png" link="/images/heir-copybara.png"
title="A diagram summarizing the copybara flow for HEIR internally to Google"
%}}

### Why bother with Copybara?

tl;dr: Automatic syncing with upstream MLIR and associated code migration.

Until HEIR has a formal governance structure in place, Google
engineers—specifically Asra Ali, Shruthi Gorantala, and Jeremy Kun—are the
codebase stewards. Because the project is young and the team is small, we want
to reduce our workload. One important aspect of that is keeping up to date with
the upstream MLIR project and incorporating bug fixes and new features into
HEIR. Google also wishes to stay up to date with MLIR and LLVM, and so it has
tooling devoted to integrating new MLIR changes into Google's monorepo every few
hours. As part of that rotation, a set of approved internal projects that depend
on MLIR (like TensorFlow) are patched to support breaking changes in MLIR. HEIR
is one of those approved projects.

As shown in the previous section, the cost of this is that no change can go into
HEIR without at least two Googlers approving it, and the project is held to a
specific set of code quality standards, namely Google's. We acknowledge these
quirks, and look forward to the day when HEIR is useful enough and important
enough that we can revisit this governance structure with the community.
