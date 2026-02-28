"""
Model configuration interactive module

Key point: model field is completely free-form, user can add any key-value pairs
"""

from prompt_toolkit import print_formatted_text, prompt


def parse_model_value(value: str):
    """
    Attempt to parse model parameter value

    Supported types:
    - int: 12
    - float: 0.1, 1e-4
    - bool: true, false
    - list: [1,2,3] or 1,2,3
    - str: other strings
    """
    value = value.strip()

    # Empty value
    if not value or value.lower() == "none":
        return None

    # Boolean
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False

    # List type
    if value.startswith("[") and value.endswith("]"):
        try:
            import ast

            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass

    # Comma-separated list
    if "," in value:
        try:
            items = []
            for item in value.split(","):
                item = item.strip()
                if "." in item or "e" in item.lower():
                    items.append(float(item))
                else:
                    try:
                        items.append(int(item))
                    except ValueError:
                        items.append(item)
            return items
        except (ValueError, SyntaxError):
            pass

    # Numeric types
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        else:
            return int(value)
    except ValueError:
        pass

    # String type
    return value


def prompt_model_params():
    """
    Interactive prompt for model parameters (free-form fields)

    Logic:
    1. Ask user if they need to add model parameters
    2. Allow user to dynamically add multiple key-value pairs
    3. Attempt intelligent type parsing for parameter values

    Returns:
        dict: Model configuration dictionary (may be empty)
    """
    print_formatted_text("\n" + "=" * 60)
    print_formatted_text("[ Model Configuration ]")
    print_formatted_text("=" * 60)
    print_formatted_text("💡 The 'model' field is completely free-form.")
    print_formatted_text("   Add any parameters needed for your model initialization.\n")

    params = {}

    while True:
        # Check if user wants to add parameters
        if params:
            print_formatted_text(f"\nCurrent model parameters: {list(params.keys())}")

        add_more = prompt("Add/modify model parameter? (yes/no, default: no): ").strip().lower()
        if add_more in ("y", "yes"):
            pass  # Continue adding
        elif add_more in ("", "n", "no"):
            break
        else:
            print_formatted_text("❌ Invalid input. Please enter 'yes' or 'no'.")
            continue

        # Input parameter name
        while True:
            param_name = prompt("  Parameter Name: ").strip()
            if param_name:
                break
            print_formatted_text("  ❌ Parameter name cannot be empty.")

        # Input parameter value
        while True:
            param_value_str = prompt("  Parameter Value: ").strip()
            try:
                param_value = parse_model_value(param_value_str)
                params[param_name] = param_value
                print_formatted_text(f"  ✓ {param_name}: {param_value} (type: {type(param_value).__name__})")
                break
            except Exception as e:
                print_formatted_text(f"  ❌ Error parsing value: {e}. Try again.")

    return params


if __name__ == "__main__":
    config = prompt_model_params()
    print("\nFinal Model Configuration:")
    import json

    print(json.dumps(config, indent=2, default=str))
