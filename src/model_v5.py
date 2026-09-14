import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyLuauGPTv5(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=256,
        n_embd=320,
        n_head=8,
        n_layer=8,
        dropout=0.1,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer

        # Token embedding
        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embd
        )

        # Position embedding
        self.position_embedding = nn.Embedding(
            block_size,
            n_embd
        )

        self.dropout = nn.Dropout(dropout)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=4 * n_embd,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layer,
        )

        self.ln_f = nn.LayerNorm(n_embd)

        # LM head
        self.lm_head = nn.Linear(
            n_embd,
            vocab_size,
            bias=False,
        )

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    def forward(self, idx, targets=None):
        B, T = idx.shape

        if T > self.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"block size {self.block_size}"
            )

        positions = torch.arange(
            0,
            T,
            device=idx.device,
        )

        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(positions)

        x = tok_emb + pos_emb
        x = self.dropout(x)

        # Causal attention mask
        mask = torch.triu(
            torch.ones(
                T,
                T,
                device=idx.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        x = self.transformer(
            x,
            mask=mask,
        )

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=40,
        stop_token_id=None,
    ):
        self.eval()

        for _ in range(max_new_tokens):

            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]

            if temperature <= 0:
                next_token = torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True,
                )

            else:
                logits = logits / temperature

                if top_k is not None:
                    k = min(
                        top_k,
                        logits.size(-1),
                    )

                    values, _ = torch.topk(
                        logits,
                        k,
                    )

                    min_value = values[:, -1].unsqueeze(-1)

                    logits = torch.where(
                        logits < min_value,
                        torch.full_like(
                            logits,
                            float("-inf"),
                        ),
                        logits,
                    )

                probabilities = F.softmax(
                    logits,
                    dim=-1,
                )

                next_token = torch.multinomial(
                    probabilities,
                    num_samples=1,
                )

            idx = torch.cat(
                [idx, next_token],
                dim=1,
            )

            if (
                stop_token_id is not None
                and next_token.item() == stop_token_id
            ):
                break

        return idx

    def num_parameters(self):
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )


if __name__ == "__main__":
    print("=" * 60)
    print("TinyLuauGPT v5 Model Test")
    print("=" * 60)

    vocab_size = 384

    model = TinyLuauGPTv5(
        vocab_size=vocab_size,
        block_size=256,
        n_embd=320,
        n_head=8,
        n_layer=8,
    )

    print()
    print(
        f"Parameters: "
        f"{model.num_parameters():,}"
    )

    x = torch.randint(
        0,
        vocab_size,
        (2, 64),
    )

    logits, loss = model(
        x,
        x,
    )

    print(
        f"Input shape:  {tuple(x.shape)}"
    )

    print(
        f"Output shape: {tuple(logits.shape)}"
    )

    print(
        f"Test loss:    {loss.item():.4f}"
    )

    print()
    print("Model test OK!")
