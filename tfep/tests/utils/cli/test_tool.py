#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Test objects and function in the module ``tfep.utils.cli.tool``.
"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import os

import pytest

from tfep.utils.cli.tool import CLITool, KeyValueOption, FlagOption, AbsolutePathOption


# =============================================================================
# TEST UTILITIES
# =============================================================================

def check_command(cmd, executable, subprogram=None, args=None, flags=None, kwargs=None):
    """Compares commands where keyword arguments can be in different order."""
    # Default arguments.
    if args is None:
        args = []
    if flags is None:
        flags = []
    if kwargs is None:
        kwargs = {}

    # Check executable and args are in the first and last positions.
    assert cmd[0] == executable
    if len(args) > 0:
        assert cmd[-len(args):] == args

    # Check subprogram.
    if subprogram is not None:
        assert cmd[1] == subprogram
        first_kwarg_idx = 2
    else:
        first_kwarg_idx = 1

    # Make a copy of the command with only flags and kwargs.
    if len(args) > 0:
        cmd = cmd[first_kwarg_idx:-len(args)]
    else:
        cmd = cmd[first_kwarg_idx:]

    # Search flags.
    for flag in flags:
        flag_idx = cmd.index(flag)
        del cmd[flag_idx]

    # For each kwarg, search its position and check
    # that the next one is the correct value.
    for k, v in kwargs.items():
        k_idx = cmd.index(k)
        assert cmd.pop(k_idx + 1) == v
        del cmd[k_idx]

    # Check that all arguments have been found.
    assert len(cmd) == 0


# =============================================================================
# TESTS
# =============================================================================

def test_unknown_kwarg():
    """Passing an undefined kwarg raises an exception."""
    class MyTool(CLITool):
        EXECUTABLE_PATH = 'mytool'
        mypar = KeyValueOption('-p')

    with pytest.raises(AttributeError, match='undefined'):
        MyTool(undefined=5)


def test_executable_subprogram():
    """Setting the SUBPROGRAM class variable result in the correct command."""
    class MyTool(CLITool):
        EXECUTABLE_PATH = 'mytool'
        SUBPROGRAM = 'subprog'
        mypar = KeyValueOption('-p')

    cmd = MyTool('orderedarg', mypar='keywordarg')
    subprocess_cmd = cmd.to_subprocess()

    print(subprocess_cmd)

    check_command(
        subprocess_cmd,
        executable='mytool',
        subprogram='subprog',
        args=['orderedarg'],
        kwargs={'-p': 'keywordarg'}
    )


def test_key_value_option_tool():
    """The ``KeyValueOption`` descriptor converts values to strings.

    Also checks that when the option is not given, it is not added to the command.

    """
    class Grep(CLITool):
        EXECUTABLE_PATH = 'grep'
        patterns_file_path = KeyValueOption('-f')
        max_count = KeyValueOption('-m')
        not_passed = KeyValueOption('-n')

    grep_cmd = Grep('input.txt', 2, max_count=3,
                    patterns_file_path='my_patterns.txt')
    subprocess_cmd = grep_cmd.to_subprocess()
    check_command(
        subprocess_cmd,
        executable='grep',
        args=['input.txt', '2'],
        kwargs={'-f': 'my_patterns.txt', '-m': '3'}
    )


def test_absolute_path_option():
    """``AbsolutePathOption`` converts relative paths to absolute ones."""
    class Grep(CLITool):
        EXECUTABLE_PATH = 'grep'
        patterns_file_path = AbsolutePathOption('-f')

    file_name = 'my_patterns.txt'
    grep_cmd = Grep('input.txt', 'input2.txt', patterns_file_path=file_name)

    expected_path = os.path.join(os.getcwd(), file_name)
    assert grep_cmd.patterns_file_path == expected_path


@pytest.mark.parametrize('value,expected_flags', [
    (True, ['-a', '-b', '--c']),
    (None, []),
    (False, ['-no-a', '--noooc'])
])
def test_flag_option(value, expected_flags):
    """Test the ``FlagOption`` descriptor.

    This tests that:
    - The ``prepend_to_false`` string is inserted correctly with options with
      both a single or double dash.
    - When the option to ``False`` when ``prepend_to_false`` is ``None``, the
      flag is not passed to the command.

    """
    class MyTool(CLITool):
        EXECUTABLE_PATH = 'mytool'
        prepended = FlagOption('-a', prepend_to_false='no-')
        unprepended = FlagOption('-b')
        prepended_double = FlagOption('--c', prepend_to_false='nooo')

    subprocess_cmd = MyTool(prepended=value, unprepended=value, prepended_double=value).to_subprocess()
    check_command(subprocess_cmd, executable='mytool', flags=expected_flags)


def test_flag_option_sanitization():
    """``FlagOption``s do not accept values other than booleans and ``None``."""
    class MyTool(CLITool):
        EXECUTABLE_PATH = 'mytool'
        myopt = FlagOption('-a')

    with pytest.raises(ValueError, match='boolean or None'):
        MyTool(myopt=2)
