---
hidden: true
---

# CONTRIBUTING

### Setting up the environment

#### With Rye

We use [Rye](https://rye.astral.sh/) to manage dependencies because it will automatically provision a Python environment with the expected Python version. To set it up, run:

```sh
$ ./scripts/bootstrap
```

Or [install Rye manually](https://rye.astral.sh/guide/installation/) and run:

```sh
$ rye sync --all-features
```

You can then run scripts using `rye run python script.py` or by activating the virtual environment:

```sh
# Activate the virtual environment - https://docs.python.org/3/library/venv.html#how-venvs-work
$ source .venv/bin/activate

# now you can omit the `rye run` prefix
$ python script.py
```

#### Without Rye

Alternatively if you don't want to install `Rye`, you can stick with the standard `pip` setup by ensuring you have the Python version specified in `.python-version`, create a virtual environment however you desire and then install dependencies using this command:

```sh
$ pip install -r requirements-dev.lock
```

### Adding and running examples

All files in the `examples/` directory can be freely edited or added to.

```py
# add an example to examples/<your-example>.py

#!/usr/bin/env -S rye run python
…
```

```sh
$ chmod +x examples/<your-example>.py
# run the example against your api
$ ./examples/<your-example>.py
```

### Using the repository from source

If you’d like to use the repository from source, you can either install from git or link to a cloned repository:

To install via git:

```sh
$ pip install git+ssh://git@github.com/LayerLens/stratix-python.git
```

Alternatively, you can build from source and install the wheel file:

Building this package will create two files in the `dist/` directory, a `.tar.gz` containing the source files and a `.whl` that can be used to install the package efficiently.

To create a distributable version of the library, all you have to do is run this command:

```sh
$ rye build
# or
$ python -m build
```

Then to install:

```sh
$ pip install ./path-to-wheel-file.whl
```

### Running tests

To run tests:

```sh
$ ./scripts/test
```

### Linting and formatting

This repository uses [ruff](https://github.com/astral-sh/ruff) and [black](https://github.com/psf/black) to format the code in the repository.

To lint:

```sh
$ ./scripts/lint
```

To format and fix all ruff issues automatically:

```sh
$ ./scripts/format
```

## Release Process

This section outlines the complete process for releasing a new version of the Stratix Python SDK.

### Step-by-Step Release Process

#### 1. Prepare the Release Branch

```sh
# Switch to the main branch and pull latest changes
$ git checkout main
$ git pull origin main

# Create or switch to the release branch
$ git checkout release
$ git pull origin release

# rebase | cherry-pick latest changes from main into release
$ git rebase main
```

#### 2. Bump the Version

Update the version number in the following files:

- **`src/layerlens/_version.py`**: Update the `__version__` string
- **`pyproject.toml`**: Update the `version` field

**Version Format**: Use semantic versioning (MAJOR.MINOR.PATCH)

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

**Examples**:

- `1.0.2` → `1.0.3` (patch release for bug fixes)
- `1.0.2` → `1.1.0` (minor release for new features)
- `1.0.2` → `2.0.0` (major release for breaking changes)

```sh
# Example: Update to version 1.0.3
# Edit src/layerlens/_version.py
__version__ = "1.0.3"

# Edit pyproject.toml
version = "1.0.3"
```

#### 3. Test and Validate

```sh
# Run all tests to ensure everything works
$ ./scripts/test

# Run linting to ensure code quality
$ ./scripts/lint

# Build the package to verify it builds correctly
$ rye build
```

#### 4. Commit and Push Changes

```sh
# Add and commit the version bump
$ git add src/layerlens/_version.py pyproject.toml
$ git commit -m "chore: bump version to 1.0.3"

# Push the release branch
$ git push origin release
```

#### 5. Create the Release Tag

Use the GitHub Actions workflow to create and push the release tag:

1. **Go to GitHub Actions**: Navigate to the "Actions" tab in the GitHub repository
2. **Find "Create Release Tag"**: Look for the workflow in the left sidebar
3. **Run workflow**: Click "Run workflow" with these settings:
   - **Use workflow from**: `release` (branch)
   - **Run in dry-run mode**: Check this box first to preview
   - **Confirm release**: Leave empty for dry-run

4. **Review dry-run output**: Check the workflow output to ensure everything looks correct

5. **Create actual release**: Run the workflow again with:
   - **Run in dry-run mode**: Uncheck this box
   - **Confirm release**: Type `YES` exactly

Alternatively, if any issue occurs, you can create the tag manually:

```sh
# Create and push the tag manually (if preferred)
$ git tag v1.0.3
$ git push origin v1.0.3
```

#### 6. Automatic Deployment

Once the tag is pushed, the "Deploy Python Package to AWS" workflow will automatically:

1. **Validate the release**: Run the validation script to ensure the tag follows semantic versioning
2. **Run tests**: Execute the full test suite
3. **Build the package**: Create distribution files
4. **Deploy to AWS**: Publish the package to the configured AWS repository

#### 7. Verify the Release

1. **Check GitHub Actions**: Ensure the deployment workflow completed successfully
2. **Verify package availability**: Check that the new version is available in your package repository
3. **Test installation**: Try installing the new version in a clean environment

### Release Checklist

Use this checklist to ensure you we don't miss any steps:

- [ ] Switched to and updated the `release` branch
- [ ] Merged latest changes from `main`
- [ ] Updated version in `src/layerlens/_version.py`
- [ ] Updated version in `pyproject.toml`
- [ ] Ran tests (`./scripts/test`)
- [ ] Ran linting (`./scripts/lint`)
- [ ] Built package successfully (`rye build`)
- [ ] Committed and pushed version bump
- [ ] Created release tag (via GitHub Actions or manually)
- [ ] Verified deployment workflow completed successfully
- [ ] Tested new version installation

### Troubleshooting

**Tag already exists**: If you need to recreate a tag, delete it first:

```sh
$ git tag -d v1.0.3           # Delete locally
$ git push origin :v1.0.3     # Delete on remote
```

**Rollback**: If you need to rollback a release, create a new patch version rather than trying to delete the problematic release.
