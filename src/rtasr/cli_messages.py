"""All the CLI presentation messages are defined here."""

ascii_art = r"""
  ___      _         _____ _         _       _   ___ ___
 | _ \__ _| |_ ___  |_   _| |_  __ _| |_    /_\ / __| _ \
 |   / _` |  _/ -_)   | | | ' \/ _` |  _|  / _ \\__ |   /
 |_|_\__,_|\__\___|   |_| |_||_\__,_|\__| /_/ \_|___|_|_\
 _________________________________________________________
 by Wordcab
"""

error_message = (
    "[bold red]Error: The {input_type} `{user_input}` is not supported.[/bold"
    " red]\n[bold"
    " red]==================================================================[/bold"
    " red]\nDo you mean one of these {input_type}s?\n"
)
