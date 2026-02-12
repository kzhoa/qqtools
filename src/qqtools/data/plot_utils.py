import matplotlib.pyplot as plt
import numpy as np


def plot_list_histogram(data, bins=10):
    """Plot a text histogram in the terminal"""

    # Calculate histogram
    hist, edges = np.histogram(data, bins=bins)
    max_count = max(hist)

    # Scale for display
    max_width = 50
    scale = max_width / max_count if max_count > 0 else 1

    print("\n" + "=" * 60)
    print("Value Distribution Histogram")
    print("=" * 60)

    for i in range(len(hist)):
        # Bin range
        bin_range = f"[{edges[i]:.2f}, {edges[i+1]:.2f})"
        # Count
        count = hist[i]
        # Create bar
        bar = "█" * int(count * scale)
        # Output
        print(f"{bin_range:20} | {bar} {count}")

    print(f"\nTotal: {len(data)} values")
    print(f"Min: {min(data):.2f}, Max: {max(data):.2f}")
    print(f"Mean: {np.mean(data):.2f}, Std: {np.std(data):.2f}")


def plot_dict_distribution(
    data_dict, terminal_width=80, sort_by="value", show_percentage=True, show_bar=True, bar_char="█"
):
    """
    Smart dictionary distribution bar-chart with auto-sizing

    Parameters:
    - data_dict: Dict[str, numeric]
    - terminal_width: Total terminal width to fit
    - sort_by: Sorting method 'key'/'value'/'none'
    - show_percentage: Whether to show percentages
    - show_bar: Whether to show bar chart
    - bar_char: Character to use for bar chart
    """
    if not data_dict:
        print("Input dictionary is empty!")
        return

    # Data preparation
    keys = list(data_dict.keys())
    values = [float(v) for v in data_dict.values()]
    total = sum(values)

    # Auto-calculate optimal column widths
    max_key_len = max(len(str(k)) for k in keys)

    # Calculate value display width
    max_val = max(values)
    if max_val.is_integer():
        value_width = len(f"{int(max_val):,}")
    else:
        value_width = max(len(f"{v:.2f}") for v in values)

    # Estimate total width needed
    key_col_width = min(max_key_len, terminal_width // 3)  # Use at most 1/3 of width
    value_col_width = max(10, min(value_width + 2, 15))
    perc_col_width = 10 if show_percentage else 0
    separator_width = 3  # For " | "

    # Calculate available width for bars
    used_width = key_col_width + separator_width + value_col_width + separator_width + perc_col_width
    bar_width = max(10, terminal_width - used_width - 5)  # Leave some margin

    # Scale for bars
    scale = bar_width / max_val if max_val > 0 else 1

    # Sorting
    items = list(zip(keys, values))
    if sort_by == "key":
        items.sort(key=lambda x: x[0])
    elif sort_by == "value":
        items.sort(key=lambda x: x[1], reverse=True)

    print("\n" + "=" * terminal_width)
    print("Dictionary Value Distribution".center(terminal_width))
    print("=" * terminal_width)

    # Header
    header = f"{'Key':<{key_col_width}} | {'Value':>{value_col_width}}"
    if show_percentage:
        header += f" | {'%':>8}"
    if show_bar:
        header += f" | Distribution"
    print(header)
    print("-" * terminal_width)

    # Print each item
    for key, value in items:
        # Format key
        key_display = str(key)
        if len(key_display) > key_col_width:
            key_display = key_display[: key_col_width - 3] + "..."

        # Format value
        if value.is_integer():
            value_str = f"{int(value):,}"
        else:
            if abs(value) < 0.001 or abs(value) >= 1e6:
                value_str = f"{value:.2e}"
            elif abs(value) < 1:
                value_str = f"{value:.4f}"
            else:
                value_str = f"{value:.2f}"

        # Truncate if too long
        if len(value_str) > value_col_width:
            value_str = value_str[: value_col_width - 2] + ".."

        # Build line
        line = f"{key_display:<{key_col_width}} | {value_str:>{value_col_width}}"

        # Add percentage
        if show_percentage and total > 0:
            percentage = value / total * 100
            if percentage < 0.01:
                perc_str = f"{percentage:.3f}%"
            elif percentage < 0.1:
                perc_str = f"{percentage:.2f}%"
            elif percentage < 1:
                perc_str = f"{percentage:.2f}%"
            else:
                perc_str = f"{percentage:.1f}%"
            line += f" | {perc_str:>8}"

        # Add bar
        if show_bar:
            bar_len = int(value * scale)
            line += f" | {bar_char * bar_len}"

        print(line)

    # Footer
    print("-" * terminal_width)
    stats_line = f"Total: {len(keys)} items, Sum: {total:,.2f}, Mean: {total/len(keys):,.2f}"
    if len(items) > 1:
        stats_line += f", Range: {max(values)-min(values):,.2f}"
    print(stats_line.center(terminal_width))


if __name__ == "__main__":

    data = [1.2, 2.3, 1.5, 3.2, 2.8, 2.1, 1.9, 3.5, 2.7, 1.8, 2.2, 3.1]
    plot_list_histogram(data)
