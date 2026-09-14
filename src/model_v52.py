import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# TinyLuauGPT v5.2
# User-facing version: Beta 0.5
# ============================================================

class TinyLuauGPTv52(nn.Module):

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

        self.dropout = nn.Dropout(
            dropout
        )

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=n_embd,
            nhead=n_head,
            dim_feedforward=n_embd * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layer,
        )

        # Final normalization
        self.ln_f = nn.LayerNorm(
            n_embd
        )

        # Language model head
        self.lm_head = nn.Linear(
            n_embd,
            vocab_size,
            bias=False,
        )

        # Weight tying
        self.lm_head.weight = (
            self.token_embedding.weight
        )

        self.apply(
            self._init_weights
        )


    # ========================================================
    # Weight initialization
    # ========================================================

    def _init_weights(self, module):

        if isinstance(
            module,
            nn.Linear
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:

                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(
            module,
            nn.Embedding
        ):

            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )


    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        idx,
        targets=None,
    ):

        B, T = idx.shape

        if T > self.block_size:

            raise ValueError(
                f"Sequence length {T} "
                f"exceeds block size "
                f"{self.block_size}"
            )

        positions = torch.arange(
            T,
            device=idx.device,
        )

        tok_emb = self.token_embedding(
            idx
        )

        pos_emb = self.position_embedding(
            positions
        )

        x = tok_emb + pos_emb

        x = self.dropout(
            x
        )

        # Causal mask
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

        x = self.ln_f(
            x
        )

        logits = self.lm_head(
            x
        )

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1)
                ),
                targets.reshape(-1),
            )

        return logits, loss


    # ========================================================
    # Generation
    # ========================================================

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=200,
        temperature=0.7,
        top_k=40,
        stop_token_id=None,
    ):

        self.eval()

        for _ in range(
            max_new_tokens
        ):

            idx_cond = idx[
                :,
                -self.block_size:
            ]

            logits, _ = self(
                idx_cond
            )

            logits = logits[
                :,
                -1,
                :
            ]

            logits = logits / max(
                temperature,
                1e-5
            )

            # Top-k sampling
            if top_k is not None:

                k = min(
                    top_k,
                    logits.size(-1)
                )

                values, _ = torch.topk(
                    logits,
                    k
                )

                threshold = values[
                    :,
                    -1
                ].unsqueeze(-1)

                logits[
                    logits < threshold
                ] = float("-inf")


            probabilities = F.softmax(
                logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            idx = torch.cat(
                (
                    idx,
                    next_token
                ),
                dim=1,
            )

            if (
                stop_token_id is not None
                and next_token.item()
                == stop_token_id
            ):

                break

        return idx


    # ========================================================
    # Parameter count
    # ========================================================

    def num_parameters(self):

        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print("=" * 50)
    print("TinyLuauGPT v5.2 Model Test")
    print("User-facing version: Beta 0.5")
    print("=" * 50)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    vocab_size = 401

    model = TinyLuauGPTv52(
        vocab_size=vocab_size,
        block_size=256,
        n_embd=320,
        n_head=8,
        n_layer=8,
        dropout=0.1,
    ).to(device)

    print(
        "Parameters:",
        f"{model.num_parameters():,}"
    )

    x = torch.randint(
        0,
        vocab_size,
        (2, 64),
        device=device,
    )

    y = torch.randint(
        0,
        vocab_size,
        (2, 64),
        device=device,
    )

    logits, loss = model(
        x,
        y
    )

    print(
        "Input shape: ",
        tuple(x.shape)
    )

    print(
        "Output shape:",
        tuple(logits.shape)
    )

    print(
        "Test loss:   ",
        f"{loss.item():.3f}"
    )

    print()
    print(
        "Model test OK!"
    )