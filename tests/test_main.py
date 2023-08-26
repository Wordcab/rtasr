"""Test the __main__ module."""

import argparse
from unittest.mock import Mock, patch

from rtasr.__main__ import execute_command, main, parse_arguments


class TestCLIMain:
    """Test the __main__ module."""

    def test_parse_arguments(self) -> None:
        """Mock the argparse.ArgumentParser.parse_args method."""
        argparse.ArgumentParser.parse_args = Mock(
            return_value=Mock(command="some_command")
        )
        args = parse_arguments()
        assert args.command == "some_command"

    def test_execute_command(self) -> None:
        """Mock the command passed to the command line interface."""
        mock_func = Mock()
        mock_func.return_value.run.return_value = None

        args = Mock(func=mock_func)

        result = execute_command(args)

        assert result == 0
        mock_func.assert_called_once_with(args)
        mock_func.return_value.run.assert_called_once_with()

        result = execute_command(args="invalid")

        assert result == 1

    def test_main(self) -> None:
        """Mock the main function."""
        mock_args = Mock()
        mock_execute_command = Mock(return_value="result")

        parse_arguments = Mock(return_value=mock_args)
        execute_command = Mock(side_effect=mock_execute_command)

        with patch("rtasr.__main__.parse_arguments", parse_arguments), patch(
            "rtasr.__main__.execute_command", execute_command
        ):
            main()

        assert parse_arguments.called
        assert execute_command.called
