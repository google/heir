# HEIR website

This site uses [Hugo](https://gohugo.io/) with the
[Docsy](https://www.docsy.dev/docs/) theme.

## Local Build

In order to build the website locally, you will need to install `hugo` and
`npm`. For example, on Ubuntu: `sudo apt-get install hugo npm`

Note that `.github/workflows/docs.yml` installs a specific version of Hugo and
Node, and it is likely that your system-provided version is not exactly the
same. See the bottom of this README for instruction on how to install the exact
versions in case you run into any errors.

1. Build the markdown files from the tablegen sources:

   ```bash
   bazel query "filter('_filegroup', siblings(kind('gentbl_rule', @heir//...)))" | xargs bazel build "$@"
   ```

1. Copy the markdown files to `docs/`:

   ```bash
   ./.github/workflows/copy_tblgen_files.sh
   ```

1. Navigate to the `/docs` directory:

   ```bash
   cd docs
   ```

1. Use `npm` to install the dependencies:

   ```bash
   npm ci
   ```

1. Run `hugo`:

   ```bash
   hugo server --minify
   ```

### Matching Hugo and Node Version Exactly

The website action
[`.github/workflows/docs.yml`](https://github.com/google/heir/blob/main/.github/workflows/docs.yml)
installs a specific version of Hugo and Node, and you should check there for the
current versions to use:

1. Install the Node Version Manager (`nvm`) Follow the instructions at
   https://github.com/nvm-sh/nvm to download `nvm`. For example:

   ```bash
   wget -qO- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.0/install.sh | bash
   ```

1. Install the same Node version as used in the workflow, for example: \`\`bash
   nvm install 18

   ```

   ```

1. Install the Hugo dependencies, minimally `golang`. For example, on Ubuntu:
   `sudo apt-get install golang`

1. Download the matching release from GitHub. The release page
   (https://github.com/gohugoio/hugo/releases/tag/v0.113.0) provides several
   versions, pick the *extended* version for your system. For example:

   ```bash
   mkdir docs/hugo && cd docs/hugo
   wget https://github.com/gohugoio/hugo/releases/download/v0.113.0/hugo_extended_0.113.0_linux-amd64.tar.gz
   tar -xzvf hugo_extended_0.113.0_linux-amd64.tar.gz
   cd ../..
   ```

1. Follow the instructions above, except for the last step (running Hugo):

   ```bash
   ./hugo/hugo server --minify
   ```
