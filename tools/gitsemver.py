#!/usr/bin/env python
## A script to fetch the latest version tag from git then append how far ahead
#  the current repository is.

from __future__ import annotations
import argparse
import re
import subprocess
import sys
import traceback
from os import error, path

class Version:

  def __init__(self, string: str, ahead: int = 0,
               modified: bool = False, gitSHA: str = '') -> None:
    '''!@brief Create a new Version object

    @param string String version number '[major].[minor].[patch](-tweak)'
    @param ahead Number of commits ahead of version string
    @param modified True indicates repository has been modified
    @param gitSHA SHA of the current repository state
    '''
    self.string = string
    self.ahead = ahead
    self.modified = modified
    self.gitSHA = gitSHA

    versionList = re.search(
        r'(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)-?([a-zA-Z\d][-a-zA-Z.\d]*)?', string)
    self.major = int(versionList[1])
    self.minor = int(versionList[2])
    self.patch = int(versionList[3])
    if versionList[4]:
      self.tweak = versionList[4]
    else:
      self.tweak = ''

  def __str__(self) -> str:
    '''!@brief Get a string representation of the Version

    @returns str Version string '[major].[minor].[patch](-tweak)'
    '''
    return self.string

  def fullStr(self) -> str:
    '''!@brief Get a string representation of the Version

    @returns str Version string '[major].[minor].[patch](-tweak)+[modified][ahead].[gitSHA]'
    '''
    if self.modified:
      return f'{self.string}+~{self.ahead}.{self.gitSHA}'
    return f'{self.string}+{self.ahead}.{self.gitSHA}'

  def __ge__(self, other: Version) -> bool:
    '''!@brief Greater than or equal to comparison

    Looks at major.minor.patch only

    @param other Other Version to compare to
    @return True if self's version is higher than or equal to others, False otherwise
    '''
    if self.major < other.major:
      return False
    if self.minor < other.minor:
      return False
    if self.patch < other.patch:
      return False
    return True

def getVersion(git: str = 'git') -> Version:
  '''!@brief Get the version information from the git tags and repository state

  @param git Executable path for git
  @return Version Extracted version information. Returns 0.0.0 if no tags/not a git repo
  '''
  try:
    # Most recent tag
    cmd = [git, 'describe', '--abbrev=0', '--tags']
    string = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.DEVNULL).strip()

    # Number of commits since last tag
    cmd = [git, 'rev-list', string + '..HEAD', '--count']
    ahead = int(subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.DEVNULL).strip())

    # Current commit SHA
    cmd = [git, 'rev-parse', '--short', 'HEAD']
    gitSHA = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.DEVNULL).strip()

    # Check if repository contains any modifications
    modified = False
    cmd = [git, 'status', '-s']
    if subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.DEVNULL).strip():
      modified = True
  except Exception as e:
    return Version('0.0.0', gitSHA='None')

  return Version(string, ahead, modified, gitSHA)

def checkSemver(cmd: str, minimum: str) -> bool:
  '''!@brief Run a command, read its output for semantic version, compare to a minimum

  @param cmd Command to git i.e. ['git', '--version']
  @param minimum Required semantic version string to compare to (inclusive)
  @return bool True if outputted version is greater or equal to the minimum, false otherwise
  '''
  try:
    output = subprocess.check_output(cmd, universal_newlines=True)
  except Exception:
    print(
        f'Unable to run \'{cmd[0]}\'. Is command correctly specified?',
        file=sys.stderr)
    return False
  matches = re.search(
      r'(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)',
      output,
      flags=re.S)
  version = Version(matches.expand(r'\1.\2.\3'))

  return version >= Version(minimum)

def checkInstallations(
  git: str = None, gitConfig: bool = False, quiet: bool = False) -> None:
  '''!@brief Check that the required installations are installed and setup

  @param git Executable path for git, None will not check for git
  @param gitConfig True will check for user.name and user.email, False will not
  @param quiet True will only raise Exceptions on error, False will print when checking each installation
  '''
  if git:
    if not quiet:
      print('Checking git version')
    if not checkSemver([git, '--version'], '2.17.0'):
      raise Exception('Install git version 2.17+')

    if gitConfig:
      if not quiet:
        print('Checking git config')
      try:
        subprocess.check_call(
          [git, 'config', '--global', 'user.name'], cwd='.', stdout=subprocess.DEVNULL)
        subprocess.check_call(
          [git, 'config', '--global', 'user.email'], cwd='.', stdout=subprocess.DEVNULL)
      except Exception:
        errorStr = 'No identity for git\n'
        errorStr += '  git config --global user.name \'Your name\'\n'
        errorStr += '  git config --global user.email \'you@example.com\''
        raise Exception(errorStr)

def main() -> None:
  '''!@brief Main program entry. Fetch the latest version tag from git then append how far ahead the current repository is. Outputs console.
  '''
  # Create an arg parser menu and grab the values from the command arguments
  parser = argparse.ArgumentParser(description='Fetch the latest version tag '
                                   'from git then append how far ahead the '
                                   'current repository is.')
  parser.add_argument('--git', metavar='PATH', default='git',
                      help='path to git binary')
  parser.add_argument('--output-str', metavar='FORMAT',
                      help='output version to stdout using the format: %%M major, %%m minor, %%p patch, %%t tweak, %%a ahead, %%~ modified, %%s SHA')
  parser.add_argument('--quiet', action='store_true', default=False,
                      help='only output return codes and errors')

  args = parser.parse_args(sys.argv[1:])
  if args.output_str:
    args.quiet = True

  checkInstallations(
      git=args.git,
      quiet=args.quiet)

  version = getVersion(args.git)

  if args.output_str:
    buf = args.output_str

    buf = re.sub(r'%M', f'{version.major}', buf)
    buf = re.sub(r'%m', f'{version.minor}', buf)
    buf = re.sub(r'%p', f'{version.patch}', buf)
    buf = re.sub(r'%t', f'{version.tweak}', buf)
    buf = re.sub(r'%a', f'{version.ahead}', buf)
    if version.modified:
      buf = re.sub(r'%~', '~', buf)
    else:
      buf = re.sub(r'%~', '', buf)
    buf = re.sub(r'%s', f'{version.gitSHA}', buf)
    print(buf)
  else:
    print(version.fullStr())


if __name__ == '__main__':
  main()
