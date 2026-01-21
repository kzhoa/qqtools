def get_param_stats(model):
    def format_memory_size(num_bytes):
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if abs(num_bytes) < 1024.0:
                return f"{num_bytes:.2f} {unit}"
            num_bytes /= 1024.0
        return f"{num_bytes:.2f} PB"

    def format_params_count(num_params):
        if num_params < 1e6:
            return f"{num_params:,}"
        elif num_params < 1e9:
            return f"{num_params / 1e6:.2f}M"
        else:
            return f"{num_params / 1e9:.2f}B"

    total_cnt = sum(p.numel() for p in model.parameters())
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    trainable_cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_bytes = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)

    total_cnt = format_params_count(total_cnt)
    total_bytes = format_memory_size(total_bytes)

    trainable_cnt = format_params_count(trainable_cnt)
    trainable_bytes = format_memory_size(trainable_bytes)

    msg = {"total": (total_cnt, total_bytes), "trainable": (trainable_cnt, trainable_bytes)}
    return msg


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
