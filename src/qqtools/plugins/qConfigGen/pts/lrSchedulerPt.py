"""
Learning rate scheduler and warmup configuration interactive module

Key point: Display corresponding parameters dynamically based on scheduler type,
avoid prompting for irrelevant fields
"""

import qqtools as qt
from prompt_toolkit import print_formatted_text, prompt
from prompt_toolkit.completion import WordCompleter

# Supported schedulers in qConfig specification
LR_SCHEDULER_PARAMS = {
    "cosine": ["T_max", "eta_min"],
    "step": ["step_size", "gamma"],
    "plateau": ["mode", "factor", "patience", "threshold", "min_lr"],
    "multi_step": ["milestones", "gamma"],
    "lambda": ["lr_lambda"],
}

LR_SCHEDULER_DEFAULTS = {
    "global": {
        "gamma": 0.1,
        "eta_min": 1.0e-6,
        "min_lr": 1.0e-6,
        "factor": 0.8,
        "patience": 10,
        "threshold": 1.0e-4,
    },
    "cosine": {
        "T_max": 100,
    },
    "step": {
        "step_size": 30,
    },
    "plateau": {
        "mode": "min",
    },
    "multi_step": {
        "milestones": [30, 60, 90],
    },
    "lambda": {
        "lr_lambda": "lambda epoch: 0.95 ** epoch",
    },
}

# Scheduler descriptions
SCHEDULER_DESCRIPTIONS = {
    "cosine": "Cosine annealing: lr decays from initial value to eta_min within T_max steps",
    "step": "Step decay: lr multiplies by gamma every step_size steps",
    "plateau": "Reduce LR on plateau: lr multiplies by factor when metric plateaus for patience steps",
    "multi_step": "Multi-step decay: adjust lr at specified milestone steps",
    "lambda": "Custom decay: lr multiplies by lambda function return value",
}

WARMUP_DEFAULTS = {
    "warmup_steps": 0,
    "warmup_factor": 0.01,
}


def parse_scheduler_input(value: str, default_value) -> any:
    """Parse scheduler parameter input"""
    if default_value is None:
        # lambda function as string
        return value.strip()

    if isinstance(default_value, bool):
        return value.lower() == "true"

    if isinstance(default_value, (tuple, list)):
        return [float(x.strip()) for x in value.split(",")]

    if isinstance(default_value, float):
        return float(value)

    if isinstance(default_value, int):
        return int(value)

    return value.strip()


def prompt_lr_scheduler_params():
    """
    Interactive prompt for learning rate scheduler and warmup parameters

    Logical relationships:
    1. User chooses whether to use scheduler, if not then return empty dict
    2. Select specific scheduler type, dynamically show corresponding parameters
    3. Ask whether to use warmup, if yes then show warmup parameters

    Returns:
        dict: Configuration dict with scheduler, scheduler_params, warmup_params
    """
    print_formatted_text("\n" + "=" * 60)
    print_formatted_text("[ Learning Rate Scheduler Configuration ]")
    print_formatted_text("=" * 60)

    # Step 1: Whether to use scheduler
    while True:
        use_scheduler = prompt("Do you want LR Scheduler? (yes/no, default: yes): ").strip().lower()
        if use_scheduler in ("", "y", "yes"):
            use_scheduler = True
            break
        elif use_scheduler in ("n", "no"):
            use_scheduler = False
            break
        else:
            print_formatted_text("❌ Invalid input. Please enter 'yes' or 'no'.")

    if not use_scheduler:
        print_formatted_text("⏭️  Skipping scheduler configuration.")
        return {}

    # Step 2: Select scheduler type
    available_schedulers = list(LR_SCHEDULER_PARAMS.keys())
    print_formatted_text(f"\nAvailable schedulers: {', '.join(available_schedulers)}")

    scheduler_completer = WordCompleter(LR_SCHEDULER_PARAMS.keys())
    while True:
        scheduler = prompt("Scheduler Name (default: 'cosine'): ", completer=scheduler_completer).strip().lower()
        scheduler = scheduler or "cosine"
        if scheduler in LR_SCHEDULER_PARAMS:
            break
        print_formatted_text(f"❌ Error: '{scheduler}' is not supported.")

    # Display description for selected scheduler
    desc = SCHEDULER_DESCRIPTIONS.get(scheduler, "")
    if desc:
        print_formatted_text(f"\n📖 {desc}")

    # Step 3: Prompt parameters based on scheduler type
    default_values = qt.qDict(LR_SCHEDULER_DEFAULTS["global"]).recursive_update(
        LR_SCHEDULER_DEFAULTS.get(scheduler, {})
    )

    params = {}
    param_names = LR_SCHEDULER_PARAMS[scheduler]
    
    for param_name in param_names:
        while True:
            default_value = default_values.get(param_name)
            prompt_msg = f"  {param_name}"
            if default_value is not None:
                prompt_msg += f" (default: {default_value})"
            prompt_msg += ": "

            value = prompt(prompt_msg).strip()
            if not value and default_value is not None:
                params[param_name] = default_value
                break

            try:
                params[param_name] = parse_scheduler_input(value, default_value)
                break
            except (ValueError, SyntaxError) as e:
                print_formatted_text(f"❌ Invalid input for {param_name}: {e}. Try again.")

    result = {"scheduler": scheduler, "scheduler_params": params}

    # Step 4: Warmup configuration (independent from scheduler)
    print_formatted_text("\n[ Learning Rate Warmup Configuration ]")
    while True:
        use_warmup = prompt("Do you want LR Warmup? (yes/no, default: yes): ").strip().lower()
        if use_warmup in ("", "y", "yes"):
            use_warmup = True
            break
        elif use_warmup in ("n", "no"):
            use_warmup = False
            break
        else:
            print_formatted_text("❌ Invalid input. Please enter 'yes' or 'no'.")

    if use_warmup:
        warmup_params = {}
        
        # warmup_steps (first priority)
        while True:
            default_val = WARMUP_DEFAULTS["warmup_steps"]
            value = prompt(f"  warmup_steps (default: {default_val}, 0=disable): ").strip()
            if not value:
                warmup_params["warmup_steps"] = default_val
                break
            try:
                warmup_steps = int(value)
                warmup_params["warmup_steps"] = warmup_steps
                break
            except ValueError:
                print_formatted_text("❌ Must be an integer.")

        # Only ask for warmup_epochs when warmup_steps <= 0
        if warmup_params["warmup_steps"] <= 0:
            print_formatted_text("  💡 Tip: warmup_steps > 0 will ignore warmup_epochs. Now using warmup_epochs.")
            while True:
                value = prompt(f"  warmup_epochs (default: 0, 0=disable): ").strip()
                if not value:
                    warmup_params["warmup_epochs"] = 0
                    break
                try:
                    warmup_epochs = int(value)
                    warmup_params["warmup_epochs"] = warmup_epochs
                    break
                except ValueError:
                    print_formatted_text("❌ Must be an integer.")
        else:
            print_formatted_text(f"  ✓ warmup_steps={warmup_params['warmup_steps']}, warmup_epochs will be ignored.")

        # warmup_factor (always needed)
        while True:
            default_val = WARMUP_DEFAULTS["warmup_factor"]
            value = prompt(f"  warmup_factor (default: {default_val}): ").strip()
            if not value:
                warmup_params["warmup_factor"] = default_val
                break
            try:
                warmup_params["warmup_factor"] = float(value)
                break
            except ValueError:
                print_formatted_text("❌ Must be a float.")

        result["warmup_params"] = warmup_params

    return result
