---
title: IDE configuration
weight: 50
---

## heir-lsp

HEIR provides an LSP server that extends the MLIR LSP server with HEIR's
dialects.

Build the LSP binary, then move it to a location on your path or point your IDE
to `bazel-bin/tools/heir-lsp`.

```bash
bazel build //tools:heir-lsp
cp bazel-bin/tools/heir-lsp /usr/local/bin
```

Note that if you change any HEIR dialects, or if HEIR's dependency on MLIR
updates and the upstream MLIR has dialect changes (which happens roughly daily),
you need to rebuild `heir-lsp` for it to recognize the changes.

## clangd

Most IDE configured to use clangd can be powered from a file called
`compile_commands.json`. To generate that for HEIR, run

```shell
bazel run @hedron_compile_commands//:refresh_all
```

This will need to be regenerated when there are major `BUILD` file changes. If
you encounter errors like `*.h.inc` not found, or syntax errors inside these
files, you may need to build those targets and then re-run the `refresh_all`
command above.

Note that you will most likely also need to install the actual `clangd` language server,
e.g., `sudo apt-get install clangd` on debian/ubuntu.

## ibazel file watcher

[`ibazel`](https://github.com/bazelbuild/bazel-watcher) is a shell around
`bazel` that watches a build target for file changes and automatically rebuilds.

```bash
ibazel build //tools:heir-opt
```

## VS Code

While a wide variety of IDEs and editors can be used for HEIR development, we
currently only provide support for [VSCode](https://code.visualstudio.com/).

### Setup

For the best experience, we recommend following these steps:

- Install the
  [MLIR](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-mlir),
  [clangd](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd)
  and
  [Bazel](https://marketplace.visualstudio.com/items?itemName=BazelBuild.vscode-bazel)
  extensions

- Install and rename Buildifier:

  You can download the latest Buildifier release, e.g., for linux-amd64 (see the
  [Bazel Release Page](https://github.com/bazelbuild/buildtools/releases/latest/)
  for a list of available binaries):

  ```bash
  wget -c https://github.com/bazelbuild/buildtools/releases/latest/download/buildifier-linux-amd64
  mv buildifier-linux-amd64 buildifier
  chmod +x buildifier
  ```

  Just as with bazel, you will want to move this somewhere on your PATH, e.g.:

  ```bash
  mkdir -p ~/bin
  echo 'export PATH=$PATH:~/bin' >> ~/.bashrc
  mv buildifier ~/bin/buildifier
  ```

  VS Code should automatically detect buildifier. If this is not successful, you
  can manually set the "Buildifier Executable" setting for the Bazel extension
  (`bazel.buildifierExecutable`).

- Disable the
  [C/C++ (aka 'cpptools')](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools)
  extension (either completely, or in the current workspace).

- Add the following snippet to your VS Code user settings found in
  .vscode/settings.json to enable autocomplete based on the
  compile_commands.json file (see above).

  ```json
    "clangd.arguments": [
      "--compile-commands-dir=${workspaceFolder}/",
      "--completion-style=detailed",
      "--query-driver=**"
    ],
  ```

- For Python formatting, HEIR uses [pyink](https://github.com/google/pyink) for
  autoformatting, which is a fork of the more commonly used
  [black](https://github.com/psf/black) formatter with some patches to support
  Google's internal style guide. To use it in VSCode, install `pyink` along with
  other python utilities needed for HEIR: `pip install -r requirements.txt` and
  install the
  [Black Formatter](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)
  extension, then add the following to your VSCode user settings
  (`.vscode/settings.json`):

  ```json
  "[python]": {
      "editor.defaultFormatter": "ms-python.black-formatter"
  },
  "black-formatter.path": [
      "path/to/pyink"
  ]
  ```

- It might be necessary to add the path to your buildifier to VSCode, though it
  should be auto-detected.

  - Open the heir folder in VSCode
  - Go to 'Settings' and set it on the 'Workspace'
  - Search for "Bazel Buildifier Executable"
  - Once you find it, write `[home-directory]/bin/buildifier ` for your specific
    \[home-directory\].

### Building, Testing, Running and Debugging with VSCode

#### Building

1. Open the "Explorer" (File Overview) in the left panel.
1. Find "Bazel Build Targets" towards the bottom of the "Explorer" panel and
   click the dropdown button.
1. Unfold the heir folder
1. Right-click on "//tools" and click the "Build Package Recursively" option

#### Testing

1. Open the "Explorer" (File Overview) in the left panel.
1. Find "Bazel Build Targets" towards the bottom of the "Explorer" panel and
   click the dropdown button.
1. Unfold the heir folder
1. Right-click on "//test" and click the "Test Package Recursively" option

#### Running and Debugging

1. Create a `launch.json` file in the `.vscode` folder, changing the `"name"`
   and `"args"` as required:

   ```json
   {
       "version": "0.2.0",
       "configurations": [
           {
               "name": "Debug Secret->BGV",
               "preLaunchTask": "build",
               "type": "lldb",
               "request": "launch",
               "program": "${workspaceFolder}/bazel-bin/tools/heir-opt",
               "args": [
                   "--secret-to-bgv",
                   "--debug",
                   "${workspaceFolder}/tests/secret_to_bgv/ops.mlir"
               ],
               "relativePathBase": "${workspaceFolder}",
               "sourceMap": {
                   "proc/self/cwd": "${workspaceFolder}",
                   "/proc/self/cwd": "${workspaceFolder}"
               }
           },
       ]
   }
   ```

   You can add as many different configurations as necessary.

1. Add Breakpoints to your program as desired.

1. Open the Run/Debug panel on the left, select the desired configuration and
   run/debug it.

- Note that you might have to hit "Enter" to proceed past the Bazel build. It
  might take several seconds between hitting "Enter" and the debug terminal
  opening.

## Vim/Neovim

### Misc config

Filetype detection

```vim
# ftdetect/mlir.vim
au BufRead,BufNewFile *.mlir setfiletype mlir
```

[Buildifier](/docs/development/bazel/#build-file-formatting) integration

```vim
# ftplugin/bzl.vim
augroup AutoFormat
  autocmd!
  autocmd BufWritePre Neoformat buildifier
augroup END
```

LSP configuration (Neovim, using
[nvim-lspconfig](https://github.com/neovim/nvim-lspconfig))

```lua
nvim_lsp["mlir_lsp_server"].setup {
    on_attach = on_attach,
    capabilities = capabilities,
    cmd = { "heir-lsp" },
}
```

Tree-sitter configuration for relevant project languages

```lua
require('nvim-treesitter.configs').setup {
  ensure_installed = {
    "markdown_inline",  -- for markdown in tablegen
    "mlir",
    "tablegen",
    "verilog",  -- for yosys
  },

  -- <... other config options ...>
}
```

Telescope-alternate config (quickly jump between cc, header, and tablegen files)

```lua
require('telescope-alternate').setup({
  mappings = {
    {
      pattern = '**/(.*).h',
      targets = {
        { template = '**/[1].cc',       label = 'cc',       enable_new = false },
        { template = '**/[1].cpp',      label = 'cpp',      enable_new = false },
        { template = '**/[1]Test.cc',   label = 'cc Test',  enable_new = false },
        { template = '**/[1]Test.cpp',  label = 'cpp Test', enable_new = false },
        { template = '**/[1].td',       label = 'tablegen', enable_new = false },
      }
    },
    {
      pattern = '**/(.*).cc',
      targets = {
        { template = '**/[1].h',        label = 'header',   enable_new = false },
        { template = '**/[1].cpp',      label = 'cpp',      enable_new = false },
        { template = '**/[1]Test.cc',   label = 'cc Test',  enable_new = false },
        { template = '**/[1]Test.cpp',  label = 'cpp Test', enable_new = false },
        { template = '**/[1].td',       label = 'tablegen', enable_new = false },
      }
    },
    {
      pattern = '**/(.*).cpp',
      targets = {
        { template = '**/[1].h',        label = 'header',   enable_new = false },
        { template = '**/[1].cc',       label = 'cc',       enable_new = false },
        { template = '**/[1]Test.cc',   label = 'test',     enable_new = false },
        { template = '**/[1]Test.cpp',  label = 'test',     enable_new = false },
        { template = '**/[1].td',       label = 'tablegen', enable_new = false },
      }
    },
    {
      pattern = '**/(.*).td',
      targets = {
        { template = '**/[1].h',        label = 'header',   enable_new = false },
        { template = '**/[1].cc',       label = 'cc',       enable_new = false },
        { template = '**/[1].cpp',      label = 'cpp',      enable_new = false },
      }
    },
    {
      pattern = '(.*)Test.(.*)',
      targets = {
        { template = '**/[1].[2]', label = 'implementation', enable_new = false },
      }
    },
  },
  open_only_one_with = 'vertical_split',
})
```

### Useful mappings

Navigate to the bazel build target for current file

```lua
vim.keymap.set('n', '<leader>eb', function()
  -- expand("%:p:h") gets the current filepath
  local buildfile = vim.fn.expand("%:p:h") .. "/BUILD"
  -- expand("%:t") gets the current filename with suffix.
  local target = vim.fn.expand("%:t")
  vim.api.nvim_command("botright vsplit " .. buildfile)
  vim.cmd("normal /" .. target .. vim.api.nvim_replace_termcodes("<CR>", true, true, true))
  vim.cmd("normal zz")
end,
  { noremap = true })
```

Set include guards according to HEIR style guide.

```lua
local function build_include_guard()
  -- project relative filepath
  local abs_path = vim.fn.expand("%")
  local rel_path = vim.fn.fnamemodify(abs_path, ":~:.")

  -- screaming case
  local upper = string.upper(rel_path)
  -- underscore separated
  local underscored = string.gsub(upper, "[./]", "_")
  -- trailing underscore
  return underscored .. "_"
end

-- mnemonic: fi = fix include (guard)
vim.keymap.set('n', '<leader>fi', function()
  local buf = vim.api.nvim_get_current_buf()
  local include_guard = build_include_guard()
  local ifndef = "#ifndef " .. include_guard
  local define = "#define " .. include_guard
  local endif = "#endif  // " .. include_guard

  vim.api.nvim_buf_set_lines(buf, 0, 2, false, { ifndef, define })
  vim.api.nvim_buf_set_lines(buf, -2, -1, false, { endif })
end, { noremap = true })
```
