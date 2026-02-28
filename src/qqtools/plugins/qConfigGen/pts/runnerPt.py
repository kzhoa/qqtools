"""
Training runner configuration interactive module

Key point: Dynamically show different parameters based on run_mode (epoch/step),
considering logical relationships
"""

from prompt_toolkit import print_formatted_text, prompt
from prompt_toolkit.completion import WordCompleter

from .earlystopPt import prompt_early_stop

RUN_MODE_DEFAULTS = {
    "epoch": {
        "run_mode": "epoch",
        "max_epochs": 100,
        "eval_interval": 1,
        "clip_grad": None,
    },
    "step": {
        "run_mode": "step",
        "max_steps": 100000,
        "eval_interval": 5000,
        "clip_grad": None,
    },
}


def prompt_runner_params():
    """
    Interactive prompt for training runner parameters

    Logical relationships:
    1. Select run mode (epoch/step), dynamically show corresponding parameters
       - epoch mode: requires max_epochs; eval_interval is in epochs
       - step mode: requires max_steps; eval_interval is in steps
    2. eval_interval meaning changes based on run_mode
    3. Prompt clip_grad (optional)
    4. Prompt early_stop (optional)

    Returns:
        dict: Training config dict with run_mode, max_epochs/max_steps, eval_interval,
              clip_grad, save_interval, early_stop parameters
    """
    print_formatted_text("\n" + "=" * 60)
    print_formatted_text("[ Training Runner Configuration ]")
    print_formatted_text("=" * 60)

    params = {}

    # Step 1: Select run mode
    mode_completer = WordCompleter(["epoch", "step"])
    while True:
        run_mode = prompt("Run Mode [epoch/step] (default: epoch): ", completer=mode_completer).strip().lower()
        if not run_mode:
            run_mode = "epoch"
        if run_mode in ("epoch", "step"):
            params["run_mode"] = run_mode
            break
        print_formatted_text("❌ Invalid input. Must be 'epoch' or 'step'.")

    # Step 2: Based on run_mode, prompt different training boundary parameters
    if params["run_mode"] == "epoch":
        print_formatted_text("\n📍 Epoch Mode: Training stops after max_epochs OR early stopping")

        # max_epochs (required)
        while True:
            default_val = RUN_MODE_DEFAULTS["epoch"]["max_epochs"]
            value = prompt(f"  Max Epochs (required, default: {default_val}): ").strip()
            if not value:
                params["max_epochs"] = default_val
                break
            try:
                epochs = int(value)
                if epochs < 1:
                    print_formatted_text("❌ max_epochs must be >= 1.")
                    continue
                params["max_epochs"] = epochs
                break
            except ValueError:
                print_formatted_text("❌ Must be an integer.")

        # max_steps (optional, for dual limit)
        while True:
            value = prompt("  Max Steps (optional, for dual limit, press Enter to skip): ").strip()
            if not value or value.lower() == "none":
                break
            try:
                steps = int(value)
                if steps < 1:
                    print_formatted_text("❌ max_steps must be >= 1.")
                    continue
                params["max_steps"] = steps
                break
            except ValueError:
                print_formatted_text("❌ Must be an integer.")

        # eval_interval (in epochs)
        print_formatted_text("  💡 Tip: eval_interval will be counted in EPOCHS")
        while True:
            default_val = RUN_MODE_DEFAULTS["epoch"]["eval_interval"]
            value = prompt(f"  Eval Interval (epochs, default: {default_val}): ").strip()
            if not value:
                params["eval_interval"] = default_val
                break
            try:
                interval = int(value)
                if interval < 1:
                    print_formatted_text("❌ eval_interval must be >= 1.")
                    continue
                params["eval_interval"] = interval
                break
            except ValueError:
                print_formatted_text("❌ Must be an integer.")

    else:  # step mode
        print_formatted_text("\n📍 Step Mode: Training stops after max_steps OR early stopping")

        # max_steps (required)
        while True:
            default_val = RUN_MODE_DEFAULTS["step"]["max_steps"]
            value = prompt(f"  Max Steps (required, default: {default_val}): ").strip()
            if not value:
                params["max_steps"] = default_val
                break
            try:
                steps = int(value)
                if steps < 1:
                    print_formatted_text("❌ max_steps must be >= 1.")
                    continue
                params["max_steps"] = steps
                break
            except ValueError:
                print_formatted_text("❌ Must be an integer.")

        # max_epochs (optional, for dual limit)
        while True:
            value = prompt("  Max Epochs (optional, for dual limit, press Enter to skip): ").strip()
            if not value or value.lower() == "none":
                break
            try:
                epochs = int(value)
                if epochs < 1:
                    print_formatted_text("❌ max_epochs must be >= 1.")
                    continue
                params["max_epochs"] = epochs
                break
            except ValueError:
                print_formatted_text("❌ Must be an integer.")

        # eval_interval (in steps)
        print_formatted_text("  💡 Tip: eval_interval will be counted in STEPS")
        while True:
            default_val = RUN_MODE_DEFAULTS["step"]["eval_interval"]
            value = prompt(f"  Eval Interval (steps, default: {default_val}): ").strip()
            if not value:
                params["eval_interval"] = default_val
                break
            try:
                interval = int(value)
                if interval < 1:
                    print_formatted_text("❌ eval_interval must be >= 1.")
                    continue
                params["eval_interval"] = interval
                break
            except ValueError:
                print_formatted_text("❌ Must be an integer.")

    # Step 3: clip_grad (optional)
    while True:
        value = prompt("\nClip Grad Norm (optional, press Enter to skip): ").strip()
        if not value or value.lower() == "none":
            break
        try:
            clip_grad = float(value)
            if clip_grad < 0.1:
                print_formatted_text("⚠️  Warning: clip_grad should be >= 0.1. Continue? (y/n)")
                ans = prompt("").strip().lower()
                if ans in ("y", "yes"):
                    params["clip_grad"] = clip_grad
                    break
            else:
                params["clip_grad"] = clip_grad
                break
        except ValueError:
            print_formatted_text("❌ Must be a float.")

    # Step 4: save_interval (optional, only counted in steps)
    while True:
        value = prompt("Save Interval in Steps (optional, press Enter to skip): ").strip()
        if not value or value.lower() == "none":
            break
        try:
            interval = int(value)
            if interval < 1:
                print_formatted_text("❌ save_interval must be >= 1.")
                continue
            params["save_interval"] = interval
            break
        except ValueError:
            print_formatted_text("❌ Must be an integer.")

    # Step 5: Early stopping configuration
    early_stop_config = prompt_early_stop()
    if early_stop_config:
        params["early_stop"] = early_stop_config

    return params
