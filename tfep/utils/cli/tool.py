#!/usr/bin/env python


# =============================================================================
# MODULE DOCSTRING
# =============================================================================

"""
Utility classes to wrap command line tools.

The module provides a class :class:`.CLITool` that provides boilerplate code to
wrap command line tools and make them compatible to :class:`~tfep.utils.cli.Launcher`.

"""


# =============================================================================
# GLOBAL IMPORTS
# =============================================================================

import abc
import inspect
import os


# =============================================================================
# CLITOOL
# =============================================================================

class CLITool:
    """Command line tool wrapper.

    The class mainly fulfills two roles:

    1. Encapsulates input and outputs of a command and provide a command
       specification that can be understood by :class:`tfep.utils.cli.Launcher`.
    2. Converts and sanitizes Python types to string command line parameters.
    3. Provides CLI interfaces with readable parameter names avoiding abbreviations
       that makes the code harder to read.

    Wrapping a new command line tool requires creating a new class that inherits
    from ``CLITool`` and defines its arguments using the options descriptors such
    as :class:`.AbsolutePathOption` and :class:`.FlagOption` (see examples below).

    The constructor takes as input ordered and keyword arguments. Keyword arguments
    must match those defined with the option descriptors when the wrapper is declared.
    Ordered arguments must be strings are appended to the command as strings.

    The path to the executable (or simply the executable name if it is in the
    system path) can be set globally through the class variable ``EXECUTABLE_PATH``,
    or it can be specific to the command instance as specified in the constructor.

    To associate a command to a particular subprogram, you can use the
    ``SUBPROGRAM`` class variable. E.g., for the gmx program in the GROMACS suite,
    creating ``CLITool`` that prepare a ``gmx mdrun ...`` command requires
     setting ``SUBPROGRAM = 'mdrun'``.

    Once defined and instantiated, a command can be run either using a
    :class:`~tfep.utils.cli.Launcher` class or the standard module ``subprocess``
    after building the command with the :func:`.CLITool.to_subprocess` method.

    Parameters
    ----------
    executable_path : str, optional
        The executable path associated to the instance of the command. If this
        is not specified, the ``EXECUTABLE_PATH`` class variable is used instead.

    See Also
    --------
    `tfep.utils.cli.Launcher` : Launch and run commands.

    Examples
    --------

    Suppose we want to create a wrapper for a subset of the command ``grep``
    that supports reading the pattern from a file. We can create a wrapper
    with the following syntax

    >>> class MyGrep(CLITool):
    ...     EXECUTABLE_PATH = 'grep'
    ...     patterns_file_path = KeyValueOption('-f')
    ...     max_count = KeyValueOption('-m')
    ...     print_version = FlagOption('-v')

    You can then create an command instance specifying the options. For example,
    :class:`.FlagOption`s takes either ``True`` or ``False``.

    >>> my_grep_cmd = MyGrep(print_version=True)

    You can then pass the command to a :class:`~tfep.utils.cli.Launcher` or use
    the :func:`.CLITool.to_subprocess` method can be used to convert the command
    to a sanitized ``list`` that can be executed by the Python standard module
    ``subprocess``.

    >>> my_grep_cmd.to_subprocess()
    ['grep', '-v']

    Another example more complex example

    >>> my_grep_cmd = MyGrep('input.txt', patterns_file_path='my_patterns.txt', max_count=3)
    >>> my_grep_cmd.to_subprocess()
    ['grep', '-f' 'my_patterns.txt', '-m', '3', 'input.txt']

    """

    SUBPROGRAM = None

    def __init__(self, *args, executable_path=None, **kwargs):
        self.args = args
        self._executable_path = executable_path

        # Check that keyword arguments match.
        options_descriptions = self._get_defined_options()
        for k, v in kwargs.items():
            if k not in options_descriptions:
                raise AttributeError('Undefined CLI option ' + k)

            # Set the value.
            setattr(self, k, v)

    @property
    def executable_path(self):
        """The path to the command executable to run."""
        if self._executable_path is None:
            return self.EXECUTABLE_PATH
        return self._executable_path

    @executable_path.setter
    def executable_path(self, value):
        self._executable_path = value

    def to_subprocess(self):
        """Convert the command to a list that can be run with the ``subprocess`` module.

        Returns
        -------
        subprocess_cmd : List[str]
            The command in subprocess format. For example ``['grep', '-v']``.

        """
        subprocess_cmd = [self.executable_path]

        # Add subprogram
        if self.SUBPROGRAM is not None:
            subprocess_cmd.append(self.SUBPROGRAM)

        # Append all options.
        for option_descriptor in self._get_defined_options().values():
            subprocess_cmd.extend(option_descriptor.to_subprocess(self))

        # Append all ordered args.
        subprocess_cmd.extend([str(x) for x in self.args])

        return subprocess_cmd

    @classmethod
    def _get_defined_options(cls):
        """Return a dict attribute_name -> description object for all CLIOptions defined."""
        options_descriptors = {}
        for attribute_name, descriptor_object in inspect.getmembers(cls, inspect.isdatadescriptor):
            if isinstance(descriptor_object, CLIOption):
                options_descriptors[attribute_name] = descriptor_object
        return options_descriptors

# =============================================================================
# CLI OPTIONS
# =============================================================================

class CLIOption(abc.ABC):
    """Generic descriptor for command line option.

    This must be inherited by all options for :class:``.CLITool`` to automatically
    discover the option. To implement this, it is sufficient to provide an
    implementation of the ``to_subprocess()`` method, which takes the object
    instance as input and outputs a list with the strings to append to the
    command in ``subprocess`` format.

    Parameters
    ----------
    option_name : str
        The name of the option in the command line interface (e.g., ``'-o'``).

    """
    def __init__(self, option_name):
        self.option_name = option_name

    def __set_name__(self, owner_type, name):
        self.public_name = name
        self.private_name = '_' + name

    def __get__(self, owner_instance, owner_type):
        if owner_instance is None:
            # This was call from the owner class. Return the descriptor object.
            return self
        return getattr(owner_instance, self.private_name)

    def __set__(self, owner_instance, value):
        setattr(owner_instance, self.private_name, value)

    @abc.abstractmethod
    def to_subprocess(self, owner_instance):
        """Return the strings to append to the command in ``subprocess`` format.

        For example, it might return something like ``['-o', 'path_to_my_file.txt']``.
        """
        pass


class KeyValueOption(CLIOption):
    """A generic command line key-value option.

    This descriptor simply converts the value to string.

    Parameters
    ----------
    option_name : str
        The name of the option in the command line interface (e.g., ``'-o'``).
    """

    def to_subprocess(self, owner_instance):
        """Implements ``CLIOption.to_subprocess()``."""
        value = getattr(owner_instance, self.private_name, None)
        if value is None:
            return []
        return [self.option_name, str(value)]


class AbsolutePathOption(KeyValueOption):
    """A file or directory path that is converted to an absolute path when instantiated.

    Relative file paths change change with the current working directory. This
    option type converts relative paths to absolute paths when the option is
    assigned so that it refers to the same file even if the working directory
    is changed.

    Parameters
    ----------
    option_name : str
        The name of the option in the command line interface (e.g., ``'-o'``).

    """
    def __set__(self, owner_instance, value):
        abs_path = os.path.abspath(value)
        setattr(owner_instance, self.private_name, abs_path)


class FlagOption(CLIOption):
    """A generic command line flag option.

    This descriptor accepts only ``True``/``False`` or ``None`` and it specifies
    CLI flag parameters (i.e., that do not take a value). If ``None``, it is not
    passed to the command. If ``False``, its behavior depends on the
    ``prepend_no_to_false`` parameter (see below).

    Parameters
    ----------
    option_name : str
        The name of the option in the command line interface (e.g., ``'-o'``).
    prepend_to_false : str, optional
        If given and the descriptor is ``False``, this string (typically ``'no'``)
        is inserted into the flag passed to the command right after the dash
        character(s).

    """
    def __init__(self, option_name, prepend_to_false=None):
        super().__init__(option_name)
        self.prepend_to_false = prepend_to_false

    def __set__(self, owner_instance, value):
        if not isinstance(value, bool) and value is not None:
            raise ValueError(self.public_name + ' must be either a boolean or None')
        setattr(owner_instance, self.private_name, value)

    def to_subprocess(self, owner_instance):
        """Implements ``CLIOption.to_subprocess()``."""
        value = getattr(owner_instance, self.private_name)
        if (value is None or (
                (not value and self.prepend_to_false is None))):
            return []

        if value is True:
            return [self.option_name]

        # value is False and self.prepend_to_false is not None.
        if self.option_name.startswith('--'):
            n_dashes = 2
        else:
            n_dashes = 1
        option_name = self.option_name[:n_dashes] + self.prepend_to_false + self.option_name[n_dashes:]
        return [option_name]
