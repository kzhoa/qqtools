"""
Early stopping configuration interactive module

Key point: First ask whether to use early stopping, then show detailed parameters if yes;
provide complete parameter configuration
"""

from prompt_toolkit import print_formatted_text, prompt
from prompt_toolkit.completion import WordCompleter

EARLY_STOP_DEFAULTS = {
    "target": "val_loss",
    "patience": 15,
    "mode": "min",
    "min_delta": 1.0e-6,
}


def prompt_early_stop():
    """
    Interactive prompt for early stopping parameters

    Logical relationships:
    1. First ask whether to use early stopping, if not then return empty dict
    2. If yes then show all early stopping related parameters

    Returns:
        dict: Early stopping config dict with target, patience, mode, min_delta, etc;
              returns empty dict if early stopping is not used
    """
    print_formatted_text("\n" + "=" * 60)
    print_formatted_text("[ Early Stopping Configuration ]")
    print_formatted_text("=" * 60)

    while True:
        use_earlystop = prompt("Do you want Early Stopping? (yes/no, default: yes): ").strip().lower()
        if use_earlystop in ("", "y", "yes"):
            use_earlystop = True
            break
        elif use_earlystop in ("n", "no"):
            use_earlystop = False
            break
        else:
            print_formatted_text("❌ Invalid input. Please enter 'yes' or 'no'.")

    if not use_earlystop:
        print_formatted_text("⏭️  Skipping early stopping configuration.")
        return {}

    params = {}

    # Monitor metric name
    while True:
        default_val = EARLY_STOP_DEFAULTS["target"]
        value = prompt(f"  Monitor Metric (default: {default_val}): ").strip()
        if not value:
            params["target"] = default_val
            break
        params["target"] = value
        break

    # Optimization direction
    while True:
        default_val = EARLY_STOP_DEFAULTS["mode"]
        mode_completer = WordCompleter(["min", "max"])
        value = (
            prompt(f"  Optimization Direction [min/max] (default: {default_val}): ", completer=mode_completer)
            .strip()
            .lower()
        )
        if not value:
            params["mode"] = default_val
            break
        if value in ("min", "max"):
            params["mode"] = value
            break
        print_formatted_text("❌ Invalid input. Must be 'min' or 'max'.")

    # Patience - wait rounds
    while True:
        default_val = EARLY_STOP_DEFAULTS["patience"]
        value = prompt(f"  Patience (default: {default_val}): ").strip()
        if not value:
            params["patience"] = default_val
            break
        try:
            patience = int(value)
            if patience < 1:
                print_formatted_text("❌ Patience must be >= 1.")
                continue
            params["patience"] = patience
            break
        except ValueError:
            print_formatted_text("❌ Must be an integer.")

    # Min delta - improvement threshold
    while True:
        default_val = EARLY_STOP_DEFAULTS["min_delta"]
        value = prompt(f"  Min Delta (improvement threshold, default: {default_val}): ").strip()
        if not value:
            params["min_delta"] = default_val
            break
        try:
            params["min_delta"] = float(value)
            break
        except ValueError:
            print_formatted_text("❌ Must be a float.")

    # Lower bound (optional)
    while True:
        value = prompt("  Lower Bound (stop if metric < this, optional, press Enter to skip): ").strip()
        if not value or value.lower() == "none":
            break
        try:
            params["lower_bound"] = float(value)
            break
        except ValueError:
            print_formatted_text("❌ Must be a float or empty.")

    # Upper bound (optional)
    while True:
        value = prompt("  Upper Bound (stop if metric > this, optional, press Enter to skip): ").strip()
        if not value or value.lower() == "none":
            break
        try:
            params["upper_bound"] = float(value)
            break
        except ValueError:
            print_formatted_text("❌ Must be a float or empty.")

    return params
