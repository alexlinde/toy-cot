import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any

def compute_weighted_loss(
    logits: torch.Tensor,
    target_tokens: torch.Tensor,
    rat_mask: torch.Tensor,
    ans_mask: torch.Tensor,
    aux_outputs: Dict[str, Any],
    aux_labels: List[Dict[str, Dict[str, int]]],
    tokenizer,
    loss_weights: Dict[str, float],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared loss: rationale, answer (masked CE), and auxiliary count heads.

    Returns total, rationale_loss, answer_loss, aux_loss.
    """
    V = tokenizer.get_vocab_size()
    ce = F.cross_entropy(
        logits.reshape(-1, V),
        target_tokens.reshape(-1),
        ignore_index=tokenizer.pad_token_id,
        reduction='none'
    ).view(target_tokens.size(0), -1)

    def masked_mean(x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        denom = m.sum().clamp_min(1.0)
        return (x * m).sum() / denom

    rationale_loss = masked_mean(ce, rat_mask)
    answer_loss    = masked_mean(ce, ans_mask)

    aux_loss = torch.tensor(0.0, device=logits.device)
    num_heads = 0
    if aux_outputs is not None:
        # (aux_outputs key, aux_labels key) pairs for each count-head family
        head_families = [
            ('count_logits', 'counts'),
            ('size_count_logits', 'size_counts'),
            ('color_count_logits', 'color_counts'),
        ]
        for out_key, label_key in head_families:
            if out_key not in aux_outputs:
                continue
            for name, head_logits in aux_outputs[out_key].items():
                targets = torch.tensor([al[label_key][name] for al in aux_labels],
                                       device=head_logits.device, dtype=torch.long)
                aux_loss = aux_loss + F.cross_entropy(head_logits, targets)
                num_heads += 1
        if num_heads > 0:
            aux_loss = aux_loss / num_heads

    total = (loss_weights['rationale'] * rationale_loss +
             loss_weights['answer']    * answer_loss +
             loss_weights['aux']       * aux_loss)

    return total, rationale_loss, answer_loss, aux_loss
