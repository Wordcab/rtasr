"""Tests for the CLI messages module."""

from rtasr.cli_messages import ascii_art, error_message


class TestCLIMessages:
    """Test the CLI messages module."""

    def test_rtasr_ascii_art(self) -> None:
        """Test the rtasr ascii art."""

    assert (
        ascii_art
        == r"""
  ___      _         _____ _         _       _   ___ ___
 | _ \__ _| |_ ___  |_   _| |_  __ _| |_    /_\ / __| _ \
 |   / _` |  _/ -_)   | | | ' \/ _` |  _|  / _ \\__ |   /
 |_|_\__,_|\__\___|   |_| |_||_\__,_|\__| /_/ \_|___|_|_\
 _________________________________________________________
 by Wordcab
"""
    )

    def test_error_message_template(self) -> None:
        """Test the error message template."""
        assert (
            error_message
            == "[bold red]Error: The {input_type} `{user_input}` is not"
            " supported.[/bold red]\n[bold"
            " red]==================================================================[/bold"
            " red]\nDo you mean one of these {input_type}s?\n"
        )
