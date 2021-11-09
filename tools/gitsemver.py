#!/usr/bin/env python
"""Reads the sematic version from git tags
Fetches the closest tag to the current state then appends number of commits
ahead the current state is.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys


class Version:
  """Semantic version object
  Stores a major, minor, and patch number; a tweak string (e.g. -rc0); number of
  commits ahead of version tag; boolean state of modification; and a SHA string
  of repository.
  Class works for just generic semantic versions and the additional information
  from the git repository.
  Attributes:
    major: integer major number (incremented on non-backwards compatible change)
    minor: integer minor number (incremented on backwards compatible change)
    patch: integer patch number (incremented on bug fix)
    tweak: string build identified (such as -RC0, release candidate 0)
    string: "[major].[minor].[patch](-tweak)"
    ahead: integer number of commits ahead
    modified: boolean modified state
    sha: SHA string, unique state identifier
  """

  def __init__(self,
               string: str,
               ahead: int = 0,
               modified: bool = False,
               sha: str = "") -> None:
    """Initialize Version with version numbers and other information
    Args:
      string: String version number "[major].[minor].[patch](-tweak)"
      ahead: Number of commits ahead of version string
      modified: True indicated repository has been modified
      sha: SHA string of the current repository state
    """
    self.ahead = ahead
    self.modified = modified
    self.sha = sha

    version_list = re.search(
        r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)-?([a-zA-Z\d][-a-zA-Z.\d]*)?",
        string)
    self.major = int(version_list[1])
    self.minor = int(version_list[2])
    self.patch = int(version_list[3])
    if version_list[4]:
      self.tweak = version_list[4]
    else:
      self.tweak = ""
    self.string = f"{self.major}.{self.minor}.{self.patch}{self.tweak}"

  def __str__(self) -> str:
    """Get a string representation of the Version
    Returns:
      A string "[major].[minor].[patch](-tweak)"
    """
    return self.string

  def full_str(self) -> str:
    """Get a string representation of the Version
    Returns:
      A string "[major].[minor].[patch](-tweak)+[modified][ahead].[gitSHA]"
    """
    if self.modified:
      return f"{self.string}+~{self.ahead}.{self.sha}"
    return f"{self.string}+{self.ahead}.{self.sha}"

  def __ge__(self, other: Version) -> bool:
    """Greater than or equal to comparison
    Looks at major.minor.patch only
    Args:
      other: Other Version to compare to
    Returns:
      True if the self's version is higher than or equal to other's
    """
    if self.major < other.major:
      return False
    if self.minor < other.minor:
      return False
    if self.patch < other.patch:
      return False
    return True


def get_version(git: str = "git") -> Version:
  """Get the version information from the git tags and repository state
  Args:
    git: Executable path for git
  Returns:
    Version extracted from git. Returns 0.0.0 if no tags/not a git repo
  """
  try:
    # Most recent tag
    cmd = [git, "describe", "--abbrev=0", "--tags"]
    string = subprocess.check_output(cmd,
                                     universal_newlines=True,
                                     stderr=subprocess.DEVNULL).strip()

    # Number of commits since last tag
    cmd = [git, "rev-list", string + "..HEAD", "--count"]
    ahead = int(
        subprocess.check_output(cmd,
                                universal_newlines=True,
                                stderr=subprocess.DEVNULL).strip())

    # Current commit SHA
    cmd = [git, "rev-parse", "--short", "HEAD"]
    sha = subprocess.check_output(cmd,
                                  universal_newlines=True,
                                  stderr=subprocess.DEVNULL).strip()

    # Check if repository contains any modifications
    modified = False
    cmd = [git, "status", "-s"]
    if subprocess.check_output(cmd,
                               universal_newlines=True,
                               stderr=subprocess.DEVNULL).strip():
      modified = True
  except Exception:  # pylint: disable=broad-except
    return Version("0.0.0", sha="None")

  return Version(string, ahead, modified, sha)


def check_semver(cmd: str, minimum: str) -> bool:
  """Run a command, read its output for semantic version, compare to a minimum
  Args:
    cmd: Command to run, i.e. ["git", "--version"]
    minimum: Required semantic version string to compare to (inclusive)
  Returns:
    True if outputted version is greater than or equal to the minimum
  """
  try:
    output = subprocess.check_output(cmd, universal_newlines=True)
  except Exception:  # pylint: disable=broad-except
    print(f"Unable to run '{cmd[0]}'. Is command correctly specified?",
          file=sys.stderr)
    return False
  matches = re.search(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)",
                      output,
                      flags=re.S)
  version = Version(matches.expand(r"\1.\2.\3"))

  return version >= Version(minimum)


def check_installations(git: str = None,
                        git_config: bool = False,
                        verbose: bool = True) -> bool:
  """Check that the required installations are installed and setup
  Args:
    git: Executable path for git, None will not check for git
    git_config: True will check for user.name and user.email, False will not
    verbose: True will print statuses along the way, False will not
  Returns:
    False when an installation does not meet its criterion, True otherwise
  """
  if git:
    if verbose:
      print("Checking git version")
    if not check_semver([git, "--version"], "2.17.0"):
      print("Install git version 2.17+", file=sys.stderr)
      return False

    if git_config:
      if verbose:
        print("Checking git config")
      try:
        subprocess.check_call([git, "config", "user.name"],
                              cwd=".",
                              stdout=subprocess.DEVNULL)
        subprocess.check_call([git, "config", "user.email"],
                              cwd=".",
                              stdout=subprocess.DEVNULL)
      except Exception:  # pylint: disable=broad-except
        print("No identity for git", file=sys.stderr)
        print('  git config --global user.name "Your name"\n', file=sys.stderr)
        print('  git config --global user.email "you@example.com"',
              file=sys.stderr)
        return False


def main() -> None:
  """Fetch the latest version tag from git then append how far ahead the
  current repository is.
  """
  # Create an arg parser menu and grab the values from the command arguments
  parser = argparse.ArgumentParser(description="Fetch the latest version tag "
                                   "from git then append how far ahead the "
                                   "current repository is.")
  parser.add_argument("--git",
                      metavar="PATH",
                      default="git",
                      help="path to git binary")
  parser.add_argument(
      "--output-str",
      metavar="FORMAT",
      help="output version to stdout using the format: %%M major, %%m minor, "
      "%%p patch, %%t tweak, %%a ahead, %%~ modified, %%s SHA")
  parser.add_argument("--verbose",
                      "-v",
                      action="store_true",
                      default=False,
                      help="output additional information")

  args = parser.parse_args(sys.argv[1:])
  if args.output_str:
    args.verbose = False

  check_installations(git=args.git, verbose=args.verbose)

  version = get_version(args.git)

  if args.output_str:
    buf = args.output_str

    buf = re.sub(r"%M", f"{version.major}", buf)
    buf = re.sub(r"%m", f"{version.minor}", buf)
    buf = re.sub(r"%p", f"{version.patch}", buf)
    buf = re.sub(r"%t", f"{version.tweak}", buf)
    buf = re.sub(r"%a", f"{version.ahead}", buf)
    if version.modified:
      buf = re.sub(r"%~", "~", buf)
    else:
      buf = re.sub(r"%~", "", buf)
    buf = re.sub(r"%s", f"{version.sha}", buf)
    print(buf)
  else:
    print(version.full_str())


if __name__ == "__main__":
  main()
