"""Global configuration interactive module: seed, log_dir, print_freq, ckp_file, init_file, render_type"""

import os

from prompt_toolkit import print_formatted_text, prompt
from prompt_toolkit.completion import WordCompleter

GLOBAL_DEFAULTS = {
    "seed": 42,
    "log_dir": "./logs/experiment",
    "print_freq": 10,
    "ckp_file": None,
    "init_file": None,
    "render_type": "auto",
}

# Render type descriptions
RENDER_TYPE_DESCRIPTIONS = {
    "auto": "Automatically choose based on environment (tqdm if available, otherwise plain text)",
    "plain": "Plain text output (no progress bar)",
    "tqdm": "Use tqdm library for progress bar display",
    "rich": "Use rich library for colored and styled output with progress bar (requires rich package)",
}


def parse_global_input(param_name: str, value: str, default_value) -> any:
    """Parse global parameter input"""
    if not value.strip():
        return default_value

    if param_name == "seed":
        return int(value)
    elif param_name == "print_freq":
        return int(value)
    elif param_name == "log_dir":
        return value.strip()
    elif param_name == "ckp_file":
        val = value.strip()
        return val if val.lower() != "none" else None

    return value.strip()


def prompt_global_params():
    """
    Interactive prompt for global configuration parameters

    Returns:
        dict: Configuration dict containing seed, log_dir, print_freq, ckp_file, init_file, render_type
    """
    print_formatted_text("\n" + "=" * 60)
    print_formatted_text("[ Global Configuration ]")
    print_formatted_text("=" * 60)

    params = {}

    # Seed
    while True:
        default_value = GLOBAL_DEFAULTS["seed"]
        value = prompt(f"Seed (default: {default_value}): ").strip()
        if not value:
            params["seed"] = default_value
            break
        try:
            params["seed"] = int(value)
            break
        except ValueError:
            print_formatted_text("❌ Invalid input. Must be an integer.")

    # Log directory
    while True:
        default_value = GLOBAL_DEFAULTS["log_dir"]
        value = prompt(f"Log Directory (default: {default_value}): ").strip()
        if not value:
            params["log_dir"] = default_value
            break
        params["log_dir"] = value
        break

    # Print frequency
    while True:
        default_value = GLOBAL_DEFAULTS["print_freq"]
        value = prompt(f"Print Frequency (batches, default: {default_value}): ").strip()
        if not value:
            params["print_freq"] = default_value
            break
        try:
            params["print_freq"] = int(value)
            break
        except ValueError:
            print_formatted_text("❌ Invalid input. Must be an integer.")

    # Checkpoint file (optional)
    while True:
        value = prompt("Checkpoint File (optional, press Enter to skip): ").strip()
        if not value or value.lower() == "none":
            break
        elif os.path.exists(value):
            params["ckp_file"] = value
            break
        else:
            print_formatted_text(f"⚠️  Warning: File '{value}' does not exist. Continue anyway? (y/n)")
            ans = prompt("").strip().lower()
            if ans in ("y", "yes"):
                params["ckp_file"] = value
                break

    # Init file (optional)
    while True:
        value = prompt("Init File (optional, press Enter to skip): ").strip()
        if not value or value.lower() == "none":
            break
        elif os.path.exists(value):
            params["init_file"] = value
            break
        else:
            print_formatted_text(f"⚠️  Warning: File '{value}' does not exist. Continue anyway? (y/n)")
            ans = prompt("").strip().lower()
            if ans in ("y", "yes"):
                params["init_file"] = value
                break

    # Render type
    print_formatted_text("\n📺 Render Type (controls log and progress bar display):")
    available_types = list(RENDER_TYPE_DESCRIPTIONS.keys())
    for rtype in available_types:
        print_formatted_text(f"  • {rtype}: {RENDER_TYPE_DESCRIPTIONS[rtype]}")

    render_completer = WordCompleter(available_types)
    while True:
        default_value = GLOBAL_DEFAULTS["render_type"]
        value = prompt(f"Render Type (default: {default_value}): ", completer=render_completer).strip().lower()
        if not value:
            params["render_type"] = default_value
            break
        if value in available_types:
            params["render_type"] = value
            break
        print_formatted_text(f"❌ Invalid render type. Choose from: {', '.join(available_types)}")

    return params


if __name__ == "__main__":
    config = prompt_global_params()
    print("\nFinal Configuration:")
    import json

    print(json.dumps(config, indent=2))
