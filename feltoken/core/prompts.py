"""Module for basic interactions with user."""


def yes_no_prompt(prompt: str, default: bool = False) -> bool:
    """Print yes/no question and check for answer.

    Args:
        prompt: yes/no question to ask
        default: default behavior if answer not matched

    Returns:
        true if user answered yes, false otherwise
    """
    x = input(f"{prompt} {'[Y/n]' if default else '[y/N]'}: ")

    if x.lower().strip() in ["y", "ye", "yes"]:
        return True

    if x.lower().strip() in ["n", "no"]:
        return False

    return default
