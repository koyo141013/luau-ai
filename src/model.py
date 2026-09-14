import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyLuauGPT(nn.Module):
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
        self.dropout = dropout

        # -------------------------------------------------
        # Embeddings
        # -------------------------------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embd,
        )

        self.position_embedding = nn.Embedding(
            block_size,
            n_embd,
        )

        self.embedding_dropout = nn.Dropout(
            dropout
        )

        # -------------------------------------------------
        # Transformer
        # -------------------------------------------------

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

        self.final_norm = nn.LayerNorm(
            n_embd
        )

        # -------------------------------------------------
        # Language model head
        # -------------------------------------------------

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size,
            bias=False,
        )

        # Weight tying
        self.lm_head.weight = (
            self.token_embedding.weight
        )

        self.apply(self._init_weights)

    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

            if module.bias is not None:
                nn.init.zeros_(
                    module.bias
                )

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,
            )

    # -----------------------------------------------------
    # Forward
    # -----------------------------------------------------

    def forward(
        self,
        idx,
        targets=None,
    ):
        batch_size, seq_len = idx.shape

        if seq_len > self.block_size:
            raise ValueError(
                f"Sequence length {seq_len} "
                f"exceeds block size {self.block_size}"
            )

        # Token positions
        positions = torch.arange(
            seq_len,
            device=idx.device,
        )

        # Embeddings
        x = self.token_embedding(idx)

        x = x + self.position_embedding(
            positions
        )

        x = self.embedding_dropout(x)

        # -------------------------------------------------
        # Causal attention mask
        # -------------------------------------------------

        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=idx.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        # Transformer
        x = self.transformer(
            x,
            mask=causal_mask,
        )

        x = self.final_norm(x)

        logits = self.lm_head(x)

        # -------------------------------------------------
        # Loss
        # -------------------------------------------------

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )

        return logits, loss

    # -----------------------------------------------------
    # Text generation
    # -----------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=180,
        temperature=0.7,
        top_k=40,
        stop_token_id=None,
    ):
        for _ in range(max_new_tokens):

            # Keep only the latest context
            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(
                idx_cond
            )

            # Last token
            logits = logits[:, -1, :]

            # Temperature
            if temperature <= 0:
                next_token = torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True,
                )

            else:
                logits = logits / temperature

                # Top-k filtering
                if top_k is not None:
                    k = min(
                        top_k,
                        logits.size(-1),
                    )

                    values, _ = torch.topk(
                        logits,
                        k,
                    )

                    cutoff = values[:, [-1]]

                    logits = torch.where(
                        logits < cutoff,
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

            # Stop at <END>
            if stop_token_id is not None:
                if torch.all(
                    next_token == stop_token_id
                ):
                    break

        return idx

    # -----------------------------------------------------
    # Parameter count
    # -----------------------------------------------------

    def num_parameters(self):
        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )


# =========================================================
# Quick test
# =========================================================

if __name__ == "__main__":
    model = TinyLuauGPT(
        vocab_size=252,
        block_size=256,
        n_embd=320,
        n_head=8,
        n_layer=8,
    )

    print("=== TinyLuauGPT v4 Model Test ===")
    print()

    print(
        f"Parameters: "
        f"{model.num_parameters():,}"
    )

    test_input = torch.randint(
        0,
        252,
        (2, 32),
    )

    logits, loss = model(
        test_input,
        test_input,
    )

    print(
        f"Input shape:  {test_input.shape}"
    )

    print(
        f"Logits shape: {logits.shape}"
    )

    print(
        f"Test loss:    {loss.item():.4f}"
    )

    print()
    print("Model test complete!")