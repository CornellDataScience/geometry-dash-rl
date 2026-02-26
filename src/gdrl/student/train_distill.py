import torch
import torch.nn as nn
from gdrl.student.model import StudentAgent


def sanity_train_step(batch_size: int = 32):
    model = StudentAgent().cuda() if torch.cuda.is_available() else StudentAgent()
    device = next(model.parameters()).device
    frames = torch.rand(batch_size, 4, 84, 84, device=device)
    teacher_probs = torch.softmax(torch.randn(batch_size, 2, device=device), dim=-1)
    modes = torch.randint(0, 4, (batch_size,), device=device)

    kl_loss = nn.KLDivLoss(reduction="batchmean")
    ce_loss = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    action_logits, mode_logits = model(frames)
    loss = kl_loss(torch.log_softmax(action_logits, dim=-1), teacher_probs) + 0.1 * ce_loss(mode_logits, modes)

    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.detach().cpu())


if __name__ == "__main__":
    print(f"sanity_loss={sanity_train_step():.4f}")
