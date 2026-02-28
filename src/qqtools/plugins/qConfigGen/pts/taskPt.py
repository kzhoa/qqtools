"""
Task configuration interactive module: dataset, target variable, data loading parameters

Key point: When prompting data loading parameters, eval_batch_size defaults to batch_size
"""

from prompt_toolkit import print_formatted_text, prompt

DATALOADER_DEFAULTS = {
    "batch_size": 32,
    "eval_batch_size": None,  # Defaults to batch_size
    "num_workers": 4,
    "pin_memory": True,
}


def prompt_task_params():
    """
    Interactive prompt for task parameters

    Logical relationships:
    1. Prompt dataset name (may be empty)
    2. Prompt data loading parameters; eval_batch_size defaults to batch_size if not set

    Returns:
        dict: Task config dict with dataset (or None), dataloader parameters
    """
    print_formatted_text("\n" + "=" * 60)
    print_formatted_text("[ Task Configuration ]")
    print_formatted_text("=" * 60)

    params = {}

    # Dataset name: if user leaves blank, store as None so config serializes to null
    value = prompt("Dataset Name (optional, press Enter to leave unset/null): ").strip()
    params["dataset"] = value if value != "" else None

    # Data loading configuration
    print_formatted_text("\n[ DataLoader Configuration ]")
    dataloader = {}

    # batch_size (required)
    while True:
        default_val = DATALOADER_DEFAULTS["batch_size"]
        value = prompt(f"  Batch Size (required, default: {default_val}): ").strip()
        if not value:
            dataloader["batch_size"] = default_val
            break
        try:
            batch_size = int(value)
            if batch_size < 1:
                print_formatted_text("❌ batch_size must be >= 1.")
                continue
            dataloader["batch_size"] = batch_size
            break
        except ValueError:
            print_formatted_text("❌ Must be an integer.")

    # eval_batch_size (optional)
    # Logic: if user doesn't input, framework uses batch_size
    print_formatted_text(
        f"  💡 Tip: If not specified, eval_batch_size defaults to batch_size ({dataloader['batch_size']})"
    )
    while True:
        value = prompt("  Eval Batch Size (optional, press Enter to use batch_size): ").strip()
        if not value or value.lower() == "none":
            # Don't add eval_batch_size, let framework use batch_size
            break
        try:
            eval_batch_size = int(value)
            if eval_batch_size < 1:
                print_formatted_text("❌ eval_batch_size must be >= 1.")
                continue
            dataloader["eval_batch_size"] = eval_batch_size
            break
        except ValueError:
            print_formatted_text("❌ Must be an integer.")

    # num_workers
    while True:
        default_val = DATALOADER_DEFAULTS["num_workers"]
        value = prompt(f"  Num Workers (default: {default_val}): ").strip()
        if not value:
            dataloader["num_workers"] = default_val
            break
        try:
            num_workers = int(value)
            if num_workers < 0:
                print_formatted_text("❌ num_workers must be >= 0.")
                continue
            dataloader["num_workers"] = num_workers
            break
        except ValueError:
            print_formatted_text("❌ Must be an integer.")

    # pin_memory (boolean)
    while True:
        default_val = DATALOADER_DEFAULTS["pin_memory"]
        default_str = "yes" if default_val else "no"
        value = prompt(f"  Pin Memory (yes/no, default: {default_str}): ").strip().lower()
        if not value:
            dataloader["pin_memory"] = default_val
            break
        if value in ("y", "yes"):
            dataloader["pin_memory"] = True
            break
        elif value in ("n", "no"):
            dataloader["pin_memory"] = False
            break
        else:
            print_formatted_text("❌ Must be 'yes' or 'no'.")

    params["dataloader"] = dataloader
    return params
