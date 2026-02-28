"""
qConfigGen - Interactive qConfig configuration generation tool

Complete workflow:
1. Global configuration (seed, log_dir, print_freq, ckp_file, init_file, render_type)
2. Task configuration (dataset, data loading)
3. Optim configuration (loss, optimizer, scheduler, warmup, ema)
4. Model configuration (model parameters, completely free-form)
5. Runner configuration (run mode, training boundary, early stopping)
6. Save as YAML file
"""

import os

import yaml
from prompt_toolkit import print_formatted_text, prompt
from rich import pretty

import qqtools as qt

from .pts import (
    prompt_ema_params,
    prompt_global_params,
    prompt_loss_params,
    prompt_lr_scheduler_params,
    prompt_model_params,
    prompt_optimizer_params,
    prompt_runner_params,
    prompt_task_params,
)


def dump_yaml_with_gaps(config: dict, path: str, order: list | None = None, gap_lines: int = 1) -> None:
    """Dump `config` to `path`, inserting `gap_lines` blank lines between top-level sections.

    If `order` is None, keys are emitted in the `config` insertion order.
    Each top-level key is dumped as its own small YAML document to allow blank-line separation.
    """
    order = order or list(config.keys())
    # Group all non-section top-level fields into a single toplevel block,
    # then emit `task`, `optim`, `model`, `runner` each as their own block
    # with an empty line after each block.
    special_keys = ["task", "optim", "model", "runner"]
    # Preserve requested order but group top-level fields first
    ordered_keys = [k for k in order if k in config]

    # Build top-level aggregate (keys not in special_keys)
    top_keys = [k for k in ordered_keys if k not in special_keys]
    parts = []
    if top_keys:
        top_block = {k: config[k] for k in top_keys}
        parts.append(yaml.safe_dump(top_block, sort_keys=False, allow_unicode=True))

    # Emit each special section in the desired order if present
    for k in special_keys:
        if k in config:
            parts.append(yaml.safe_dump({k: config[k]}, sort_keys=False, allow_unicode=True))

    # Ensure there is at least one blank line between parts.
    joiner = "\n" * (gap_lines + 1)
    body = joiner.join(p.rstrip() for p in parts) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)


def prompt_save_location():
    """Prompt user for save location and filename"""
    print_formatted_text("\n" + "=" * 60)
    print_formatted_text("[ Save Configuration ]")
    print_formatted_text("=" * 60)

    cwd = os.path.abspath(os.getcwd())

    # Save directory
    while True:
        save_dir = prompt(f"Save Directory (default: {cwd}): ").strip()
        if not save_dir:
            save_dir = cwd
        if os.path.isdir(save_dir):
            break
        print_formatted_text(f"❌ Directory does not exist: {save_dir}. Try again.")

    # Filename
    while True:
        file_name = prompt("File Name (default: config.yaml): ").strip()
        if not file_name:
            file_name = "config.yaml"
        # Ensure extension
        if not file_name.endswith((".yaml", ".yml")):
            file_name += ".yaml"
        break

    return save_dir, file_name


def main():
    """Main interactive workflow"""
    pretty.install()

    print_formatted_text("\n" + "=" * 60)
    print_formatted_text("✨ Welcome to qConfigGen - qConfig Configuration Generator ✨")
    print_formatted_text("=" * 60)
    print_formatted_text("This tool will guide you through creating a complete YAML config file.\n")

    config = {}

    # Step 1: Global configuration
    print_formatted_text("\n📌 Step 1/6: Global Configuration")
    global_params = prompt_global_params()
    config.update(global_params)

    # Step 2: Task configuration
    print_formatted_text("\n📌 Step 2/6: Task Configuration")
    task_config = prompt_task_params()
    config["task"] = task_config

    # Step 3: Optimizer configuration
    print_formatted_text("\n📌 Step 3/6: Optimizer Configuration")

    # Loss
    loss_config = prompt_loss_params()

    # Optimizer
    optim_config = prompt_optimizer_params()

    # Scheduler
    scheduler_config = prompt_lr_scheduler_params()

    # EMA
    ema_config = prompt_ema_params()

    # Merge optim section
    optim_dict = qt.qDict(loss_config).recursive_update(optim_config).recursive_update(scheduler_config)
    if ema_config:
        optim_dict = optim_dict.recursive_update({"ema_params": ema_config})

    config["optim"] = optim_dict.to_dict()

    # Step 4: Model configuration
    print_formatted_text("\n📌 Step 4/6: Model Configuration")
    model_config = prompt_model_params()
    if model_config:
        config["model"] = model_config

    # Step 5: Training runner configuration (moved to later)
    print_formatted_text("\n📌 Step 5/6: Training Runner Configuration")
    runner_config = prompt_runner_params()
    config["runner"] = runner_config

    # Step 6: Save configuration
    print_formatted_text("\n📌 Step 6/6: Save Configuration")
    save_dir, file_name = prompt_save_location()

    # Save to file
    file_path = os.path.join(save_dir, file_name)
    # Use custom dump to insert blank lines between top-level sections
    # By default, respect config key order. Adjust `order` if you want a specific layout.
    dump_yaml_with_gaps(config, file_path, order=None, gap_lines=1)

    # Display summary
    print_formatted_text("\n" + "=" * 60)
    print_formatted_text("✅ Configuration saved successfully!")
    print_formatted_text("=" * 60)
    print_formatted_text(f"\n📂 File: {file_path}")
    print_formatted_text(f"\n📋 Configuration Summary:")
    print_formatted_text(f"  - Seed: {config.get('seed', 'N/A')}")
    print_formatted_text(f"  - Log Dir: {config.get('log_dir', 'N/A')}")
    print_formatted_text(f"  - Render Type: {config.get('render_type', 'N/A')}")
    print_formatted_text(f"  - Loss: {config.get('optim', {}).get('loss', 'N/A')}")
    print_formatted_text(f"  - Optimizer: {config.get('optim', {}).get('optimizer', 'N/A')}")
    print_formatted_text(f"  - Run Mode: {config.get('runner', {}).get('run_mode', 'N/A')}")
    print_formatted_text(f"  - Dataset: {config.get('task', {}).get('dataset', 'N/A')}")
    print_formatted_text(f"  - Batch Size: {config.get('task', {}).get('dataloader', {}).get('batch_size', 'N/A')}")
    if config.get("model"):
        print_formatted_text(f"  - Model Params: {list(config['model'].keys())}")
    print_formatted_text("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
