<!-- mdformat off(yaml frontmatter) -->

______________________________________________________________________

## title: Contributing to HEIR weight: 1

<!-- mdformat on -->

There are several ways to contribute to HEIR, including:

- Discussing high-level designs and questions on HEIR's
  [discussions page](https://github.com/google/heir/discussions)
- Improving or expanding HEIR's documentation
- Contributing to HEIR's [code-base](https://github.com/google/heir)
- Discuss project direction at HEIR's
  [Working Group meetings](https://google.github.io/heir/community/)

## Ways to contribute

We welcome pull requests. Please take a look at issues marked with
[good first issue](https://github.com/google/heir/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
or
[contributions welcome](https://github.com/google/heir/issues?q=is%3Aissue+is%3Aopen+label%3A%22contributions+welcome%22).

For new proposals, please open a GitHub
[issue](https://github.com/google/heir/issues) or start a
[discussion](https://github.com/google/heir/discussions) for feedback.

## Contributing to code using pull requests

### Preparing a pull request

Pre-requisites:

1. Sign the
   [Contributor License Agreement](https://cla.developers.google.com/about)
   (CLA). See more
   [here](https://github.com/google/heir/blob/main/CONTRIBUTING.md#sign-our-contributor-license-agreement).

1. Fork the HEIR repository by clicking the **Fork** button on the
   [repository page](https://github.com/google/heir). This creates a copy of the
   HEIR repository on your own GitHub account.

1. See [Getting Started](https://google.github.io/heir/docs/getting_started/) to
   install developer dependencies to build and run tests.

1. Add the HEIR repository as an upstream remote, so you can use sync your
   changes against it.

```bash
git remote add upstream https://www.github.com/google/heir
```

5. Create a development branch for your change:

```bash
git checkout -b name-of-change
```

And implement your changes using your favorite IDE. See
[IDE Configuration](https://google.github.io/heir/docs/ide_configuration/) for
more.

6. Check HEIR's lint and style checks by running the following from the top of
   the repository:

```bash
pre-commit run --all
```

7. Make sure tests are passing with the following:

```bash
bazel test @heir//...
```

8. Once you are ready with your change, create a commit as follows.

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
- Once the PR is approved, you will see a `squash ready` label applied if your
  PR requires you to squash together multiple commits. You may use the
  `git rebase -i`. Pull requests must comprise of a single git commit before
  merging.

4. **Pull Ready**

- Once the PR is squashed into a single git commit, the `pull ready` label is
  applied.
- This initiates the internal code migration and presubmits.
- If needed, we may come to you to make some minor changes in case tests fail at
  this stage. If so, we will request changes from the GitHub UI and the PR must
  be approved again.
- At times, it may not be your change, but it may be additional internal lints
  and layering checks that are not integrated with `bazel` or open-source
  tooling. We will go ahead and fix this.
- Once the internal tests pass, we merge the code internally and externally. You
  will see `copybara-bot` close the PR and merge your commit directly into main.
