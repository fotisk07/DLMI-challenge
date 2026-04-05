import torch
import copy

def save_model(model, epoch, optimizer, acc, args, name):
    if hasattr(model, "merge_and_unload"):
        model_to_save = copy.deepcopy(model).merge_and_unload()
    else:
        model_to_save = model

    torch.save(
        {
            "epoch": epoch,
            "model": model_to_save.state_dict(),
            "arch": args.arch,
            "normalizer": args.normalizer,
            "optimizer": optimizer.state_dict(),
            "best_val_acc": acc,
            "args": vars(args),
        },
        args.output_dir / f"{name}.pt",
    )