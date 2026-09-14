import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyLuauGPTv51(nn.Module):

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

        # Dropout
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

        # Final normalization
        self.ln_f = nn.LayerNorm(n_embd)

        # Language model head
        self.lm_head = nn.Linear(
            n_embd,
            vocab_size,
            bias=False
        )

        # Weight tying
        self.lm_head.weight = (
            self.token_embedding.weight
        )

        # Causal mask
        mask = torch.triu(
            torch.ones(
                block_size,
                block_size,
                dtype=torch.bool
            ),
            diagonal=1
        )

        self.register_buffer(
            "causal_mask",
            mask,
            persistent=False
        )

        self.apply(self._init_weights)


    def _init_weights(self, module):

        if isinstance(
            module,
            nn.Linear
        ):

            if module.weight is not None:

                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02
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
                std=0.02
            )


    def forward(
        self,
        idx,
        targets=None
    ):

        B, T = idx.shape

        if T > self.block_size:

            raise ValueError(
                f"Sequence length {T} "
                f"exceeds block size "
                f"{self.block_size}"
            )


        # Positions
        positions = torch.arange(
            0,
            T,
            device=idx.device
        )


        # Embeddings
        x = (
            self.token_embedding(idx)
            +
            self.position_embedding(positions)
        )

        x = self.dropout(x)


        # Causal transformer
        x = self.transformer(
            x,
            mask=self.causal_mask[
                :T,
                :T
            ]
        )


        x = self.ln_f(x)


        # Logits
        logits = self.lm_head(x)


        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1)
                ),
                targets.reshape(-1)
            )


        return logits, loss


    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens=200,
        temperature=0.7,
        top_k=40,
        stop_token_id=None,
    ):

        for _ in range(max_new_tokens):

            # Block size 제한
            idx_cond = idx[
                :,
                -self.block_size:
            ]


            output = self(
                idx_cond
            )

            logits = output[0]


            # 마지막 토큰
            logits = logits[
                :,
                -1,
                :
            ]


            # Temperature
            temperature = max(
                temperature,
                0.05
            )

            logits = (
                logits / temperature
            )


            # Top-k sampling
            if top_k is not None:

                k = min(
                    top_k,
                    logits.size(-1)
                )

                values, indices = torch.topk(
                    logits,
                    k
                )

                filtered = torch.full_like(
                    logits,
                    float("-inf")
                )

                filtered.scatter_(
                    1,
                    indices,
                    values
                )

                logits = filtered


            # Probability
            probs = F.softmax(
                logits,
                dim=-1
            )


            # Sample
            next_token = torch.multinomial(
                probs,
                num_samples=1
            )


            # Stop token
            if (
                stop_token_id is not None
                and next_token.item()
                == stop_token_id
            ):

                break


            # Append
            idx = torch.cat(
                [
                    idx,
                    next_token
                ],
                dim=1
            )


        return idx


    def num_parameters(self):

        return sum(
            p.numel()
            for p in self.parameters()
        )


def main():

    print("=" * 50)
    print("TinyLuauGPT v5.1 Model Test")
    print("=" * 50)


    vocab_size = 426

    model = TinyLuauGPTv51(
        vocab_size=vocab_size,
        block_size=256,
        n_embd=320,
        n_head=8,
        n_layer=8,
        dropout=0.1,
    )


    print(
        "Parameters:",
        model.num_parameters()
    )


    # 테스트 입력
    x = torch.randint(
        0,
        vocab_size,
        (2, 64)
    )

    targets = torch.randint(
        0,
        vocab_size,
        (2, 64)
    )


    logits, loss = model(
        x,
        targets
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
        round(loss.item(), 4)
    )


    print()
    print("Model test OK!")


if __name__ == "__main__":
    main()