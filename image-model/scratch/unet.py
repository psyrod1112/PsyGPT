import torch
import torch.nn as nn

from diffusion import sinusoidal_time_embedding


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int):
        super().__init__()

        # 1. conv1: in_channels -> out_channels, kernel_size=3, padding=1
        #    (padding=1로 해야 H,W가 안 줄어듦)
        self.conv1 = ...

        # 2. norm1: nn.GroupNorm(num_groups, out_channels)
        #    num_groups는 보통 8로 시작 (out_channels가 8로 나눠떨어져야 함, 안되면 num_groups=1도 가능)
        self.norm1 = ...

        # 3. time_mlp: time embedding([batch, time_embed_dim])을 out_channels로 projection하는 Linear
        self.time_mlp = ...

        # 4. conv2: out_channels -> out_channels, kernel_size=3, padding=1
        self.conv2 = ...

        # 5. norm2: nn.GroupNorm(num_groups, out_channels)
        self.norm2 = ...

        # 6. activation: nn.SiLU() (하나 만들어서 forward에서 재사용)
        self.act = ...

        # 7. skip_connection: in_channels != out_channels일 때만 채널 수를 맞춰주는 1x1 conv
        #    같으면 그냥 통과(nn.Identity())
        #    힌트: if in_channels != out_channels: self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        #          else: self.skip = nn.Identity()
        self.skip = ...

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x:     [batch, in_channels, H, W]
        t_emb: [batch, time_embed_dim]  (sinusoidal_time_embedding의 결과)

        returns: [batch, out_channels, H, W]
        """
        # 1. h = conv1(x) -> norm1 -> act
        h = ...

        # 2. time embedding을 out_channels로 projection하고 [batch, out_channels, 1, 1]로 reshape해서 h에 더하기
        #    힌트: time_bias = self.time_mlp(t_emb)  # [batch, out_channels]
        #          h = h + time_bias.view(...)
        time_bias = ...
        h = h + time_bias.view(-1, time_bias.shape[1], 1, 1)

        # 3. h = conv2(h) -> norm2 -> act
        h = ...

        # 4. residual 더하기: h + self.skip(x)
        out = ...

        return out


if __name__ == "__main__":
    batch = 4
    time_embed_dim = 32

    block = ResBlock(in_channels=1, out_channels=16, time_embed_dim=time_embed_dim)

    x = torch.randn(batch, 1, 28, 28)
    t = torch.tensor([0, 1, 500, 999])
    t_emb = sinusoidal_time_embedding(t, embed_dim=time_embed_dim)

    out = block(x, t_emb)
    print("입력 shape:", x.shape)
    print("출력 shape (기대: [4, 16, 28, 28]):", out.shape)

    # time embedding이 실제로 영향을 주는지 확인: 서로 다른 t를 넣으면 출력도 달라야 함
    t2 = torch.tensor([999, 999, 999, 999])
    t_emb2 = sinusoidal_time_embedding(t2, embed_dim=time_embed_dim)
    out2 = block(x, t_emb2)
    diff = (out - out2).abs().mean().item()
    print("같은 x, 다른 t로 넣었을 때 출력 차이 (0이 아니어야 함):", diff)
