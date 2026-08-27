import math
import torch


def sinusoidal_time_embedding(t: torch.Tensor, embed_dim: int) -> torch.Tensor:
    """
    t: [batch] 정수(혹은 float) timestep 텐서
    embed_dim: 임베딩 벡터 크기 (짝수 가정)

    returns: [batch, embed_dim]
    """
    half_dim = embed_dim // 2

    # 1. freqs: [half_dim] 크기 텐서. freqs[i] = 1 / (10000 ^ (i / half_dim))
    #    힌트: exponent = torch.arange(half_dim) / half_dim 을 만든 다음
    #          freqs = 10000 ** (-exponent)  (또는 torch.pow 사용)
    exponent = torch.arange(half_dim) / half_dim
    freqs = torch.pow(10000, -exponent)

    # 2. args: t와 freqs를 곱해서 [batch, half_dim] 만들기
    #    t는 [batch] shape인데 freqs([half_dim])와 곱하려면 t를 [batch, 1]로 reshape 후 브로드캐스팅
    args = t.view(-1, 1) * freqs.view(1, -1)

    # 3. sin, cos을 마지막 축(dim=-1)으로 이어붙이기 (torch.cat) -> [batch, embed_dim]
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    return embedding


class NoiseScheduler:
    """
    Forward diffusion process: x_0 -> x_t

    q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    """

    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        # 1. self.betas: beta_start에서 beta_end까지 num_timesteps개로 선형 증가하는 1차원 텐서
        #    (torch.linspace 사용)
        self.betas = torch.linspace(beta_start, beta_end, steps=num_timesteps)

        # 2. self.alphas: 1 - betas
        self.alphas = 1 - self.betas

        # 3. self.alpha_bars: alphas의 누적곱 (alpha_bar_t = alpha_1 * alpha_2 * ... * alpha_t)
        #    (torch.cumprod 사용, dim=0)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x_0: torch.Tensor, t: torch.Tensor):
        """
        x_0: [batch, C, H, W] 원본 이미지
        t:   [batch] 각 샘플마다 다른 timestep (정수 인덱스, 0 ~ num_timesteps-1)

        returns:
            x_t:   [batch, C, H, W] 노이즈가 섞인 이미지
            noise: [batch, C, H, W] 실제로 섞은 표준정규분포 노이즈 (나중에 학습 타겟으로 씀)
        """
        # 1. noise: x_0와 같은 shape의 표준정규분포 랜덤 텐서 (torch.randn_like)
        noise = torch.randn_like(x_0)

        # 2. alpha_bar_t: self.alpha_bars에서 t에 해당하는 값들을 뽑아오기 (self.alpha_bars[t])
        #    shape는 [batch] -> broadcasting 위해 [batch, 1, 1, 1]로 reshape 필요
        alpha_bar_t = self.alpha_bars[t].view(-1, 1, 1, 1)

        # 3. 공식대로 x_t 계산: sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise

        return x_t, noise


if __name__ == "__main__":
    scheduler = NoiseScheduler(num_timesteps=1000)

    x_0 = torch.randn(4, 1, 28, 28)  # 배치 4개짜리 가짜 MNIST 이미지라고 가정

    # t=0 근처: 노이즈가 거의 안 섞여서 x_t가 x_0와 거의 같아야 함
    t_small = torch.tensor([0, 0, 0, 0])
    x_t_small, _ = scheduler.add_noise(x_0, t_small)
    print("t=0일 때 x_t와 x_0 차이 (거의 0이어야 함):", (x_t_small - x_0).abs().mean().item())

    # t=999 근처: 노이즈가 거의 다 섞여서 x_t가 순수 노이즈에 가까워야 함
    t_large = torch.tensor([999, 999, 999, 999])
    x_t_large, noise = scheduler.add_noise(x_0, t_large)
    print("t=999일 때 x_t와 noise 차이 (거의 0에 가까워야 함):", (x_t_large - noise).abs().mean().item())

    print("betas[0], betas[-1]:", scheduler.betas[0].item(), scheduler.betas[-1].item())
    print("alpha_bars[0], alpha_bars[-1]:", scheduler.alpha_bars[0].item(), scheduler.alpha_bars[-1].item())

    # Lesson 2: time embedding 테스트
    t = torch.tensor([0, 1, 500, 999])
    emb = sinusoidal_time_embedding(t, embed_dim=32)
    print("\ntime embedding shape (기대: [4, 32]):", emb.shape)
    print("서로 다른 t의 embedding은 달라야 함 (t=0 vs t=999 차이):", (emb[0] - emb[3]).abs().mean().item())
    print("emb 값 범위 (sin/cos이라 -1~1 사이여야 함): min", emb.min().item(), "max", emb.max().item())
