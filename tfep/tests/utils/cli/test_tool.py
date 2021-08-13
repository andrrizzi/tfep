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
# TESTS
# =============================================================================

def test_unknown_kwarg():
    """Passing an undefined kwarg raises an exception."""
    class MyTool(CLITool):
        EXECUTABLE_PATH = 'mytool'
        mypar = KeyValueOption('-p')

    with pytest.raises(AttributeError, match='undefined'):
        MyTool(undefined=5)


def test_key_value_option_tool():
    """The ``KeyValueOption`` descriptor converts values to strings.."""
    class Grep(CLITool):
        EXECUTABLE_PATH = 'grep'
        patterns_file_path = KeyValueOption('-f')
        max_count = KeyValueOption('-m')

    grep_cmd = Grep('input.txt', 2,
                    patterns_file_path='my_patterns.txt', max_count=3)
    subprocess_cmd = grep_cmd.to_subprocess()
    assert set(subprocess_cmd) == set(['grep', '-f', 'my_patterns.txt', '-m', '3', 'input.txt', '2'])


def test_absolute_path_option():
    """``AbsolutePathOption`` converts relative paths to absolute ones."""
    class Grep(CLITool):
        EXECUTABLE_PATH = 'grep'
        patterns_file_path = AbsolutePathOption('-f')

    file_name = 'my_patterns.txt'
    grep_cmd = Grep('input.txt', 'input2.txt', patterns_file_path=file_name)

    expected_path = os.path.join(os.getcwd(), file_name)
    assert grep_cmd.patterns_file_path == expected_path


@pytest.mark.parametrize('value,expected_cmd', [
    (True, ['mytool', '-a', '-b', '--c']),
    (None, ['mytool']),
    (False, ['mytool', '-no-a', '--noooc'])
])
def test_flag_option(value, expected_cmd):
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
    assert set(subprocess_cmd) == set(expected_cmd)


def test_flag_option_sanitization():
    """``FlagOption``s do not accept values other than booleans and ``None``."""
    class MyTool(CLITool):
        EXECUTABLE_PATH = 'mytool'
        myopt = FlagOption('-a')

    with pytest.raises(ValueError, match='boolean or None'):
        MyTool(myopt=2)
