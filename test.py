import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

src_positions = (
            torch.arange(5, 15, device=device)
            .unsqueeze(1)
            .expand(10, 5)
            .to(device)
        )

print(src_positions)


