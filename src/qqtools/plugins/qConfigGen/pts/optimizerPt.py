"""
Optimizer and loss function configuration interactive module

Key point: Support single loss function, comboloss multi-target, and complete EMA configuration
"""

import qqtools as qt
from prompt_toolkit import print_formatted_text, prompt
from prompt_toolkit.completion import WordCompleter

# Supported loss functions
LOSS_FUNCTIONS = ["mse", "mae", "l1", "l2mae", "bce", "ce", "cross_entropy", "focal", "comboloss"]

# Supported optimizers and parameters
OPTIMIZER_PARAMS = {
    "adamw": ["lr", "weight_decay", "betas", "eps"],
    "adam": ["lr", "weight_decay", "betas", "eps"],
    "sgd": ["lr", "momentum", "weight_decay", "nesterov"],
    "rmsprop": ["lr", "alpha", "weight_decay", "momentum"],
}

OPTIMIZER_DEFAULTS = {
    "global": {
        "lr": 1.0e-3,
        "weight_decay": 1.0e-4,
        "eps": 1.0e-8,
    },
    "adamw": {
        "betas": (0.9, 0.999),
    },
    "adam": {
        "betas": (0.9, 0.999),
    },
    "sgd": {
        "momentum": 0.9,
        "nesterov": False,
    },
    "rmsprop": {
        "alpha": 0.99,
        "momentum": 0.0,
    },
}

EMA_DEFAULTS = {
    "ema": False,
    "ema_decay": 0.99,
}


def parse_optimizer_input(value: str, default_value) -> any:
    """Parse optimizer parameter input"""
    if default_value is None:
        return value.strip()

    if isinstance(default_value, bool):
        return value.lower() in ("true", "yes")

    if isinstance(default_value, (tuple, list)):
        return tuple(float(x.strip()) for x in value.split(","))

    if isinstance(default_value, float):
        return float(value)

    if isinstance(default_value, int):
        return int(value)

    return value.strip()


def prompt_loss_params():
    """
    Interactive prompt for loss function configuration

    Logical relationships:
    1. Select loss function type
    2. If comboloss is selected, need to input multiple targets and their parameters
    3. Otherwise just record loss function name

    Returns:
        dict: Loss function configuration containing loss and optional loss_params
    """
    print_formatted_text("\n[ Loss Function Configuration ]")
    print_formatted_text(f"Available losses: {', '.join(LOSS_FUNCTIONS)}")

    loss_completer = WordCompleter(LOSS_FUNCTIONS)
    while True:
        loss = prompt("Loss Function (default: mse): ", completer=loss_completer).strip().lower()
        loss = loss or "mse"
        if loss in LOSS_FUNCTIONS:
            break
        print_formatted_text(f"❌ Error: '{loss}' is not supported.")

    result = {"loss": loss}

    # If comboloss is selected, need to configure multiple targets
    if loss == "comboloss":
        print_formatted_text("\n  ComboLoss: Configure weighted losses for multiple targets")
        loss_params = {}

        while True:
            # Input target name (first target is required)
            while True:
                target_name = prompt("    Target Name: ").strip()
                if target_name:
                    break
                print_formatted_text("    ❌ Target name cannot be empty.")

            # Input loss function for this target
            target_loss_completer = WordCompleter([l for l in LOSS_FUNCTIONS if l != "comboloss"])
            while True:
                target_loss = (
                    prompt(f"    Loss for '{target_name}' (default: mse): ", completer=target_loss_completer)
                    .strip()
                    .lower()
                )
                target_loss = target_loss or "mse"
                if target_loss in LOSS_FUNCTIONS and target_loss != "comboloss":
                    break
                print_formatted_text(f"    ❌ Invalid loss: '{target_loss}'.")

            # Input weight
            while True:
                weight_str = prompt(f"    Weight for '{target_name}' (default: 1.0): ").strip()
                try:
                    weight = float(weight_str) if weight_str else 1.0
                    loss_params[target_name] = [target_loss, weight]
                    break
                except ValueError:
                    print_formatted_text("    ❌ Weight must be a float.")

            # Ask whether to add another target (default: no)
            while True:
                more = prompt("  Add another target? (yes/no, default: no): ").strip().lower()
                if more in ("", "n", "no"):
                    break
                if more in ("y", "yes"):
                    # continue outer loop to add next target
                    break
                print_formatted_text("  ❌ Invalid input. Please enter 'yes' or 'no'.")

            if more in ("", "n", "no"):
                # Ensure at least one target exists (it does because we just added one)
                result["loss_params"] = loss_params
                break
            # otherwise loop to add another target

    return result


def prompt_optimizer_params():
    """
    Interactive prompt for optimizer configuration

    Returns:
        dict: Optimizer configuration containing optimizer and optimizer_params
    """
    print_formatted_text("\n[ Optimizer Configuration ]")
    available_optimizers = list(OPTIMIZER_PARAMS.keys())
    print_formatted_text(f"Available optimizers: {', '.join(available_optimizers)}")

    # Select optimizer
    optimizer_completer = WordCompleter(OPTIMIZER_PARAMS.keys())
    while True:
        optimizer = prompt("Optimizer Name (default: adamw): ", completer=optimizer_completer).strip().lower()
        optimizer = optimizer or "adamw"
        if optimizer in OPTIMIZER_PARAMS:
            break
        print_formatted_text(f"❌ Error: '{optimizer}' is not supported.")

    # Get default values for this optimizer
    default_values = qt.qDict(OPTIMIZER_DEFAULTS["global"]).recursive_update(OPTIMIZER_DEFAULTS.get(optimizer, {}))

    # Input optimizer parameters
    params = {}
    for param_name in OPTIMIZER_PARAMS[optimizer]:
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
                params[param_name] = parse_optimizer_input(value, default_value)
                break
            except (ValueError, SyntaxError) as e:
                print_formatted_text(f"  ❌ Invalid input for {param_name}: {e}. Try again.")

    return {"optimizer": optimizer, "optimizer_params": params}


def prompt_ema_params():
    """
    Interactive prompt for EMA configuration

    Logical relationships:
    1. First ask whether to enable EMA
    2. Only prompt ema_decay parameter when enabled

    Returns:
        dict: EMA configuration containing ema and ema_decay; return empty dict if not enabled
    """
    print_formatted_text("\n[ EMA (Exponential Moving Average) Configuration ]")

    while True:
        use_ema = prompt("Do you want EMA? (yes/no, default: no): ").strip().lower()
        if use_ema in ("", "n", "no"):
            use_ema = False
            break
        elif use_ema in ("y", "yes"):
            use_ema = True
            break
        else:
            print_formatted_text("❌ Invalid input. Please enter 'yes' or 'no'.")

    if not use_ema:
        print_formatted_text("⏭️  Skipping EMA configuration.")
        return {}

    print_formatted_text("  📖 EMA Decay: Coefficient for exponential moving average (higher = slower change)")

    # Only ask for ema_decay when EMA is enabled
    while True:
        default_val = EMA_DEFAULTS["ema_decay"]
        value = prompt(f"  EMA Decay (default: {default_val}, range: 0-1): ").strip()
        if not value:
            ema_decay = default_val
        else:
            try:
                ema_decay = float(value)
                if not (0 <= ema_decay <= 1):
                    print_formatted_text("  ❌ EMA decay must be between 0 and 1.")
                    continue
            except ValueError:
                print_formatted_text("  ❌ Must be a float.")
                continue

        return {"ema": True, "ema_decay": ema_decay}
