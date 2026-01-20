# 💃🏼 NCSN으로 CelebA 데이터셋 훈련

본 문서는 2019년 제안된 **Noise Conditional Score Network(NCSN)** 를 PyTorch 기반으로 재구현하고, 그 실험적 동작을 평가한 결과를 기술한다. 본 구현은 score-based generative modeling의 핵심 구성 요소를 충실히 따르며, 학습 단계에서는 Annealed Denoising Score Matching(DSM)을, 샘플링 단계에서는 Annealed Langevin Dynamics(ALD)를 사용한다.

#### NCSN 샘플링 미리보기

| 초기 분포 | `seed=42` | `seed=10` |
|------|------|------|
| $\mathcal{U}(-1,1)\in\mathbb{R}^{N\times H\times W}$ | <p align="center"><img src="https://velog.velcdn.com/images/lumerico284/post/c6a43ef5-3986-4483-acf5-59e54c13fdd2/image.gif" width="70%"/></p> | <p align="center"><img src="https://velog.velcdn.com/images/lumerico284/post/e8b3a26b-a1c1-47d9-bc2f-17ccf88c20cb/image.gif" width="70%"/></p> |
| $\mathcal{N}(\mathbf{0},\mathbf{I})\in\mathbb{R}^{N\times H\times W}$ | <p align="center"><img src="https://velog.velcdn.com/images/lumerico284/post/0d8a3a83-474c-49f1-b06f-31397425ab49/image.gif" width="70%"/></p> | <p align="center"><img src="https://velog.velcdn.com/images/lumerico284/post/e6c4c7ef-9545-4802-98cf-fbd955bb52b8/image.gif" width="70%"/></p> |

본 프로젝트의 목적은 원 논문의 방법론을 실험적으로 재현하고, 이론–구현 간 대응 관계를 명확히 드러내는 코드베이스를 구축하는 데 있다.

---


## 1️⃣ 이론적 배경

Score 기반 생성 모델(score-based generative models)은 다양한 노이즈 조건에서 데이터의 로그 밀도(log-density)의 기울기(score)를 근사함으로써 복잡한 분포를 모델링하는 강력한 접근법이다. NCSN은 노이즈 조건화된 score 네트워크를 이용해 각 노이즈 레벨에서의 score를 추정하고, 이를 기반으로 Langevin dynamics를 통해 새로운 샘플을 생성한다.

### 표기(Notation)

본 문서에서는 **벡터/텐서**를 **굵은 글씨**로, **스칼라**는 일반 글씨로 표기한다.

- 데이터 샘플(이미지): $\mathbf{x}\in \mathbb{R}^{C\times H\times W}$
- 가우시안 노이즈: $\boldsymbol\epsilon\sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
- 노이즈 스케일(스칼라): $\sigma > 0$
- 이산 노이즈 레벨(정수): $y \in \{0,\ldots,K-1\}$

아래에서 $\nabla_{\mathbf{x}}$는 $\mathbf{x}$에 대한 기울기(gradient)를 의미한다.

### Score 추정 및 노이즈 조건화

데이터 분포 $p_{\text{data}}(\mathbf{x})$에 대해, NCSN은 가우시안 노이즈가 추가된

$$
\mathbf{x}_\sigma = \mathbf{x} + \sigma \mathbf{z},\quad \mathbf{z}\sim \mathcal{N}(\mathbf{0},\mathbf{I})
$$

를 사용한다. Score 네트워크 $s_\theta(\mathbf{x}_\sigma, \sigma)$는

$$
\nabla_{\mathbf{x}_\sigma}\log p_\sigma(\mathbf{x}_\sigma)
$$

를 근사하도록 학습된다. 실제 구현에서는 연속적인 $\sigma$ 대신 이산적 노이즈 레벨 $y \in \{0,\ldots,K-1\}$을 사용한다.

여기서 $p_\sigma(\mathbf{x}_\sigma)$는 데이터에 가우시안 노이즈(분산 $\sigma^2$)를 섞어 만든 주변(marginal) 분포로, 다음과 같은 커널(perturbation kernel)로 표현된다:

$$
q_\sigma(\mathbf{x}_\sigma\mid \mathbf{x})=
\mathcal{N}(\mathbf{x}_\sigma;\mathbf{x},\sigma^2\mathbf{I}),
\quad
p_\sigma(\mathbf{x}_\sigma)=
\int p_{\text{data}}(\mathbf{x})\,q_\sigma(\mathbf{x}_\sigma\mid \mathbf{x})\,d\mathbf{x}.
$$

NCSN의 목표는 다양한 $\sigma$에서의 score $\nabla_{\mathbf{x}_\sigma}\log p_\sigma(\mathbf{x}_\sigma)$를 동시에 추정하는 것이다. 이를 위해 입력을 (노이즈가 섞인) $\mathbf{x}_\sigma$와 노이즈 레벨 $y$로 두고, $y$가 가리키는 $\sigma_y$에서의 score를 출력하도록 네트워크를 조건화한다.

또한 주변 분포의 score는 **조건부 기댓값**으로도 표현된다:

$$
\nabla_{\mathbf{x}_\sigma}\log p_\sigma(\mathbf{x}_\sigma)=\mathbb{E}\left[\nabla_{\mathbf{x}_\sigma}\log q_\sigma(\mathbf{x}_\sigma\mid \mathbf{x})\mid \mathbf{x}_\sigma\right]=
\mathbb{E}\left[\frac{\mathbf{x}-\mathbf{x}_\sigma}{\sigma^2}\bigg| \mathbf{x}_\sigma\right].
$$

즉, 노이즈가 섞인 관측치 $\mathbf{x}_\sigma$가 주어졌을 때 원본 $\mathbf{x}$를 평균적으로 얼마나 되돌려야 하는가가 곧 score를 학습하는 문제로 연결된다.

#### 노이즈 조건화가 필요한 이유

큰 $\sigma$에서는 분포가 부드러워져 score가 비교적 단순해지고, 작은 $\sigma$에서는 데이터 매니폴드 근처의 복잡한 구조를 더 정교하게 복원해야 한다. 따라서 하나의 네트워크가 여러 노이즈 스케일을 처리하려면 “현재 노이즈 레벨이 무엇인지”를 명시적으로 알려주는 조건($\sigma$ 또는 $y$)이 필요하다.

### Annealed Denoising Score Matching(DSM)

DSM 학습 목적 함수는 다음과 같다:

$$
\mathbb{E}_{\mathbf{x}\sim p_{\text{data}},\,\mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})}
\left[
\left\|
s_\theta(\mathbf{x} + \sigma \mathbf{z}, \sigma) + \frac{\mathbf{z}}{\sigma}
\right\|^2
\right].
$$

수치적 안정성을 위해 이를

$$
\left\|
\sigma\cdot s_\theta(\mathbf{x} + \sigma \mathbf{z}, \sigma) + \mathbf{z}
\right\|^2
$$

형태로 재스케일링하여 사용한다.

#### Score Matching 관점

이상적으로는 각 $\sigma$에서의 true score $\nabla_{\mathbf{x}_\sigma}\log p_\sigma(\mathbf{x}_\sigma)$를 직접 맞추는 다음 목적을 생각할 수 있다:

$$
\mathbb{E}_{\mathbf{x}_\sigma\sim p_\sigma}
\left[
\left\| s_\theta(\mathbf{x}_\sigma,\sigma) - \nabla_{\mathbf{x}_\sigma}\log p_\sigma(\mathbf{x}_\sigma)\right\|^2
\right].
$$

이는 Fisher divergence(Score matching)와 연결되지만, $p_\sigma$ 자체를 모르기 때문에 $\nabla\log p_\sigma$를 직접 계산할 수 없다. DSM은 $q_\sigma(\mathbf{x}_\sigma\mid\mathbf{x})$의 성질(가우시안 커널)을 이용해 정답 타깃을 $-\mathbf{z}/\sigma$ 형태로 바꿔서, 동일하게 L2 회귀 문제로 학습 가능하게 만든다.

#### 최적해가 $-\mathbf{z}/\sigma$ 형태로 나타나는 이유

노이즈가 섞인 관측치 $\mathbf{x}_\sigma$를 고정했을 때, 다음의 conditional score identity가 성립한다:

$$
\nabla_{\mathbf{x}_\sigma}\log q_\sigma(\mathbf{x}_\sigma\mid \mathbf{x})=
-\frac{\mathbf{x}_\sigma-\mathbf{x}}{\sigma^2}.
$$

그런데 $\mathbf{x}_\sigma=\mathbf{x}+\sigma\mathbf{z}$이므로 $\mathbf{x}_\sigma-\mathbf{x}=\sigma\mathbf{z}$, 따라서

$$
\nabla_{\mathbf{x}_\sigma}\log q_\sigma(\mathbf{x}_\sigma\mid \mathbf{x})=
-\frac{\mathbf{z}}{\sigma}.
$$

DSM은 주변 분포 $p_\sigma(\mathbf{x}_\sigma)$의 score를 직접 계산하기 어려우므로, 위의 조건부 항을 이용해 score를 학습한다. 즉, $\mathbf{x}_\sigma$를 입력으로 받는 네트워크가 $-\mathbf{z}/\sigma$에 가까운 값을 내도록 유도하면, 결과적으로 $\nabla_{\mathbf{x}_\sigma}\log p_\sigma(\mathbf{x}_\sigma)$를 잘 근사하도록 학습된다는 것이 핵심 아이디어이다.

#### Annealed(다중 스케일) DSM

실제로는 단일 $\sigma$가 아니라 여러 스케일 $\{\sigma_i\}_{i=1}^{K}$를 사용한다. NCSN에서는 $y$를 샘플링하여 $\sigma=\sigma_y$를 선택하고, 그 스케일에서의 DSM 손실을 평균낸다:

$$
\mathbb{E}_{y}\;
\mathbb{E}_{\mathbf{x},\mathbf{z}}
\left[
\left\|
\sigma_y\, s_\theta(\mathbf{x}+\sigma_y\mathbf{z}, y) + \mathbf{z}
\right\|^2
\right].
$$

또한 구현 관점에서는 **score의 크기(scale)**가 $\sigma$에 따라 크게 달라지는 경향이 있어(특히 작은 $\sigma$에서 더 큰 변화가 필요), 손실에서 $\sigma$로 재스케일링(위 식)하거나 모델 출력에 $\sigma^{-1}$를 반영하는 방식으로 학습을 안정화한다.

### 노이즈 스케줄(Noise Schedule)

노이즈 레벨은 보통 log-space에서 등간격으로 배치한 뒤 지수로 되돌리는 방식(geometric progression)을 사용한다. 즉 $\sigma_{\max}=\sigma_{\text{begin}}$, $\sigma_{\min}=\sigma_{\text{end}}$에 대해

$$
\sigma_i=
\exp\Big(\log\sigma_{\max} + \frac{i}{K-1}\big(\log\sigma_{\min}-\log\sigma_{\max}\big)\Big),
\quad i=0,\ldots,K-1.
$$

이렇게 하면 큰 노이즈 구간과 작은 노이즈 구간을 모두 안정적으로 커버할 수 있고, annealing 과정에서 스케일이 다른 score를 점진적으로 활용하기가 쉬워진다. 본 구현에서는 이산 레벨 $y=i$에 대해 $\sigma=\sigma_i$를 대응시키며, 학습 시에는 배치마다 $y$를 균등 샘플링해 다양한 노이즈 레벨을 고르게 학습한다.

### Annealed Langevin Dynamics(ALD) 샘플링

고노이즈(큰 $\sigma$)에서 시작하여 점차 낮은 $\sigma$로 이동하면서 다음과 같은 Langevin 업데이트를 반복 수행한다:

$$
\mathbf{x} \leftarrow \mathbf{x} + \alpha\, s_\theta(\mathbf{x}, \sigma) + \sqrt{2\alpha}\,\boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon}\sim\mathcal{N}(\mathbf{0},\mathbf{I})
$$

스텝 크기는

$$
\alpha_i = \eta\left( \frac{\sigma_i}{\sigma_{\min}} \right)^2
$$

로 조정한다.

#### Score를 따라 확률이 높은 쪽으로 이동

Score $\nabla_{\mathbf{x}}\log p(\mathbf{x})$는 현재 위치 $\mathbf{x}$에서 로그 밀도의 증가 방향을 가리킨다. 따라서 업데이트

$$
\mathbf{x}\leftarrow \mathbf{x} + \alpha\, s_\theta(\mathbf{x},\sigma)
$$

는 (근사된) score 방향으로 이동해 더 높은 확률 질량 영역으로 샘플을 밀어넣고, $\sqrt{2\alpha}\boldsymbol{\epsilon}$ 항은 탐색을 위한 랜덤성을 제공한다. 큰 $\sigma$에서는 분포가 매끄러워 전역적인 구조를 잡기 쉽고, 작은 $\sigma$로 갈수록 세부 질감을 정교하게 보정하게 된다(annealing).

#### 초기화와 값 한정

실무적으로는 초기 $\mathbf{x}$를 **균등분포** 또는 **정규분포**에서 샘플링한다. 또한 이미지 정규화 범위가 $[-1,1]$인 경우, 업데이트 과정에서 값이 폭주하지 않도록 각 스텝마다 $\mathbf{x}$를 $[-1,1]$로 클램프하는 선택을 자주 사용한다.

#### SDE 관점

이상적인 경우(정확한 score를 안다고 가정)에는 다음의 Langevin SDE가 목표 분포를 stationary distribution으로 갖는다:

$$
d\mathbf{x}_t = \nabla_{\mathbf{x}}\log p(\mathbf{x}_t)\,dt + \sqrt{2}\,d\mathbf{w}_t,
$$

여기서 $\mathbf{w}_t$는 브라운 운동(Brownian motion)이다. 이를 오일러-마루야마(Euler–Maruyama)로 이산화하면

$$
\mathbf{x}\leftarrow \mathbf{x} + \alpha\,\nabla_{\mathbf{x}}\log p(\mathbf{x}) + \sqrt{2\alpha}\,\boldsymbol{\epsilon}
$$

을 얻고, 실제로는 $\nabla_{\mathbf{x}}\log p(\mathbf{x})$ 대신 근사치 $s_\theta(\mathbf{x},\sigma)$를 사용한다. NCSN에서는 $p(\mathbf{x})$ 대신 $p_\sigma(\mathbf{x})$를 다루며, $\sigma$를 큰 값에서 작은 값으로 점차 낮춰가며(annealing) 샘플을 *거친 구조 $\to$ 세부 구조* 순서로 정제한다.

---

## 2️⃣ 구현 세부 내용

### Conditional Instance Normalization

노이즈 조건화를 위해 다음 형태의 CIN을 사용한다:

$$
\text{CIN}(h, y) = \gamma_y \odot \text{IN}(h) + \beta_y,
$$

여기서 $\gamma_y, \beta_y$는 노이즈 레벨 $y$마다 독립적으로 학습되는 파라미터이다.

구현에서는 `InstanceNorm2d(affine=False)`로 $\text{IN}(h)$를 계산하고, `Embedding(num_classes, 2*C)`로 각 노이즈 레벨의 $(\gamma_y,\beta_y)$를 생성한다. 또한 초기 상태에서 $\gamma_y=1,\ \beta_y=0$이 되도록 임베딩 가중치를 설정해(정규화만 적용된 상태로 시작) 학습을 안정화한다.

```python
# src/model.py
class ConditionalInstanceNorm2d(nn.Module):
    def __init__(self, num_features: int, num_classes: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.num_features = num_features
        self.norm = nn.InstanceNorm2d(num_features, affine=False, eps=eps)
        self.embed = nn.Embedding(num_classes, num_features * 2)

        with torch.no_grad():
            self.embed.weight[:, :num_features].fill_(1.0)   # gamma init
            self.embed.weight[:, num_features:].zero_()      # beta init

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.dtype != torch.long:
            y = y.long()
        h = self.norm(x)
        gamma_beta = self.embed(y)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.view(-1, self.num_features, 1, 1)
        beta = beta.view(-1, self.num_features, 1, 1)
        return h * gamma + beta
```

### Score Network 구조

네트워크 백본은 RefineNet 스타일 구조를 기반으로 하며, 출력은 이론식과 일치하도록 $\sigma^{-1}$ 배율 조정이 이루어진다. 이는 노이즈 스케일 간 일관된 학습을 돕는다.

아래 구현에서 `labels=y`는 이론 파트의 이산 노이즈 레벨($y\in\{0,\dots,K-1\}$)에 해당하며, 내부의 여러 블록(예: `RCUBlock`, `RefineBlock`)은 CIN을 통해 동일한 특성 맵을 서로 다른 노이즈 조건으로 변조한다. 마지막으로 `scale_by_sigma=True`인 경우 모델 출력 `out`을 `out / sigma_y`로 나눠서, 네트워크가 $\sigma$에 따라 스케일이 다른 score를 직접 학습하는 부담을 줄인다. 이 스케일링은 아래 DSM 손실에서 다시 $\sigma$를 곱해($\sigma s_\theta(\cdot,\sigma)$) 이론식과 정확히 대응된다.

```python
# src/model.py
class NCSN(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        nf: int = 128,
        num_classes: int = 10,
        dilations: Sequence[int] = (1, 2, 4, 8),
        scale_by_sigma: bool = True,
    ) -> None:
        super().__init__()
        if len(dilations) != 4:
            raise ValueError("Expected 4 dilation values (for 4 RefineNet stages).")

        self.in_channels = in_channels
        self.nf = nf
        self.num_classes = num_classes
        self.scale_by_sigma = bool(scale_by_sigma)

        self.register_buffer("sigmas", torch.empty(num_classes), persistent=False)

        self.begin_conv = nn.Conv2d(in_channels, nf, kernel_size=3, stride=1, padding=1)

        self.stage1 = RCUBlock(nf, num_classes, num_units=2, dilation=dilations[0])
        self.stage2 = RCUBlock(nf, num_classes, num_units=2, dilation=dilations[1])
        self.stage3 = RCUBlock(nf, num_classes, num_units=2, dilation=dilations[2])
        self.stage4 = RCUBlock(nf, num_classes, num_units=2, dilation=dilations[3])

        self.refine4 = RefineBlock([nf], nf, num_classes)
        self.refine3 = RefineBlock([nf, nf], nf, num_classes)
        self.refine2 = RefineBlock([nf, nf], nf, num_classes)
        self.refine1 = RefineBlock([nf, nf], nf, num_classes)

        self.end_norm = ConditionalInstanceNorm2d(nf, num_classes)
        self.end_act = nn.ELU()
        self.end_conv = nn.Conv2d(nf, in_channels, kernel_size=3, stride=1, padding=1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels.dim() != 1:
            labels = labels.view(-1)

        h = self.begin_conv(x)
        h1 = self.stage1(h, labels)
        h2 = self.stage2(h1, labels)
        h3 = self.stage3(h2, labels)
        h4 = self.stage4(h3, labels)

        r4 = self.refine4([h4], labels)
        r3 = self.refine3([h3, r4], labels)
        r2 = self.refine2([h2, r3], labels)
        r1 = self.refine1([h1, r2], labels)

        out = self.end_conv(self.end_act(self.end_norm(r1, labels)))
        if self.scale_by_sigma:
            used_sigmas = self.sigmas[labels].view(-1, 1, 1, 1)
            out = out / used_sigmas
        return out

    @torch.no_grad()
    def set_sigmas(self, sigmas: torch.Tensor) -> None:
        if sigmas.dim() != 1:
            raise ValueError("sigmas must be 1-D.")
        if sigmas.numel() != self.num_classes:
            raise ValueError(
                f"sigmas length ({sigmas.numel()}) must match num_classes ({self.num_classes})."
            )
        self.sigmas.copy_(sigmas.to(self.sigmas.device, dtype=self.sigmas.dtype))
```

### DSM 손실 함수

학습 과정에서는 각 배치마다 무작위 노이즈 레벨과 가우시안 노이즈를 샘플링하여 DSM 손실을 계산한다. 구현은 이론식과 완전히 일치한다.

이론 파트의 재스케일링된 DSM 목적함수

$$
\left\| \sigma s_\theta(x + \sigma z, \sigma) + z \right\|^2
$$

를 그대로 계산한다. 구체적으로는 배치마다 `labels=y`를 샘플링해 $\sigma_y$를 선택하고, `perturbed = x + sigma_y * noise`로 $x+\sigma z$를 만들며, 모델이 예측한 score `score = s_\theta(perturbed, y)`에 $\sigma_y$를 곱한 뒤 노이즈 $z$와의 제곱오차를 취한다.

```python
# src/losses/dsm.py
def annealed_dsm_loss(model, x: torch.Tensor, sigmas: torch.Tensor):
    batch = x.shape[0]
    device = x.device
    labels = torch.randint(0, sigmas.shape[0], (batch,), device=device, dtype=torch.long)
    used_sigmas = sigmas[labels].view(batch, 1, 1, 1)

    noise = torch.randn_like(x)
    perturbed = x + used_sigmas * noise
    score = model(perturbed, labels)

    loss = torch.mean(torch.sum((score * used_sigmas + noise) ** 2, dim=(1, 2, 3)))
    return loss, labels
```

### Langevin 샘플링 절차

샘플러는 각 노이즈 레벨별로 여러 Langevin 스텝을 수행하며, 진행 상황은 tqdm으로 시각화된다. 샘플은 고노이즈에서 시작해 점차 저노이즈 단계로 이동하면서 생성된다.

ALD 업데이트는 코드에서 `x = x + step_size * grad + sqrt(2*step_size) * noise`로 구현되며,

$$
x \leftarrow x + \alpha s_\theta(x, \sigma) + \sqrt{2\alpha}\varepsilon
$$

스텝 크기는 `step_size = step_lr * (sigma / sigmas[-1]) ** 2`로 대응된다(여기서 `sigmas[-1]`가 $\sigma_{\min}$).

$$
\alpha_i=\eta\left(\frac{\sigma_i}{\sigma_{\min}}\right)^2
$$

```python
# src/runners/scorenet_runner.py
@torch.no_grad()
def annealed_langevin_dynamics(
    model,
    sigmas: torch.Tensor,
    n_samples: int,
    image_size: int,
    in_channels: int,
    n_steps_each: int,
    step_lr: float,
    device: torch.device,
    clamp: bool = True,
    denoise: bool = False,
    init: Optional[torch.Tensor] = None,
    init_distribution: str = "uniform",
) -> torch.Tensor:
    model.eval()
    sigmas = sigmas.to(device)

    if init is None:
        if init_distribution == "uniform":
            x = torch.empty(n_samples, in_channels, image_size, image_size, device=device).uniform_(-1.0, 1.0)
        else:
            x = torch.randn(n_samples, in_channels, image_size, image_size, device=device)
    else:
        x = init.to(device)

    for i, sigma in enumerate(sigmas):
        labels = torch.full((n_samples,), i, device=device, dtype=torch.long)
        step_size = step_lr * (sigma / sigmas[-1]) ** 2

        for _ in range(n_steps_each):
            grad = model(x, labels)
            noise = torch.randn_like(x)
            x = x + step_size * grad + torch.sqrt(2.0 * step_size) * noise
            if clamp:
                x = x.clamp(-1.0, 1.0)

    if denoise:
        last_label = torch.full((n_samples,), sigmas.shape[0] - 1, device=device, dtype=torch.long)
        x = x + (sigmas[-1] ** 2) * model(x, last_label)
        if clamp:
            x = x.clamp(-1.0, 1.0)

    return x
```

### 학습 루프

학습 루프는 배치 단위 진행 상황(loss, running loss, epoch 평균)을 모두 기록하며, 체크포인트에는 optimizer 상태 및 running loss가 포함된다.

학습 루프는 설정 파일에서 $\{\sigma_i\}_{i=1}^K$를 만들고, `annealed_dsm_loss`로 이론식의 DSM 손실을 계산하며, 주기적으로 체크포인트/샘플 이미지를 저장한다. 즉, 이론식(DSM, ALD) $\to$ 코드 함수 호출이 `train()`에서 그대로 직결된다.

```python
# src/main.py (발췌)
sigmas = make_sigmas(
    sigma_begin=float(cfg["model"].get("sigma_begin", 1.0)),
    sigma_end=float(cfg["model"].get("sigma_end", 0.01)),
    num_scales=int(cfg["model"].get("num_scales", 10)),
).to(device)

model = NCSN(
    in_channels=int(cfg["model"].get("in_channels", 3)),
    nf=int(cfg["model"].get("nf", 128)),
    num_classes=int(cfg["model"].get("num_scales", 10)),
    dilations=tuple(cfg["model"].get("dilations", (1, 2, 4, 8))),
    scale_by_sigma=bool(cfg["model"].get("scale_by_sigma", True)),
).to(device)
model.set_sigmas(sigmas)

for x in loader:
    x = x.to(device)
    optimizer.zero_grad(set_to_none=True)
    loss, _ = annealed_dsm_loss(model, x, sigmas)
    loss.backward()
    grad_clip = cfg["training"].get("grad_clip", None)
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
    optimizer.step()

samples = annealed_langevin_dynamics(
    model=model,
    sigmas=sigmas,
    n_samples=int(cfg["sampling"].get("n_samples", 64)),
    image_size=int(cfg["data"].get("image_size", 32)),
    in_channels=int(cfg["model"].get("in_channels", 3)),
    n_steps_each=int(cfg["sampling"].get("n_steps_each", 100)),
    step_lr=float(cfg["sampling"].get("step_lr", 2e-5)),
    device=device,
)
```

---

## 4️⃣ 사용 방법

#### 학습 실행

```bash
python -u src/main.py --dataset celeba --mode train
python -u src/main.py --dataset mnist --mode train
```

#### 체크포인트 재개

```bash
python -u src/main.py --dataset mnist --mode train --resume latest
```

#### 샘플링

```bash
python -u src/main.py --dataset mnist --mode sample --ckpt latest --out out/sample.png
```

#### GIF 생성

```bash
python -u src/make_sampling_gif.py --dataset mnist --frames 50 --seed 123
```

#### Loss 곡선 시각화

```bash
python -u src/plot_running_losses.py --out_dir out --ckpt latest
```

---

## 5️⃣ 실험

### 모델 구조

- 총 파라미터 수: $3,176,067$
- MACs: $3.1\times10^9$
- 순전파/역전파 크기: `43.63 MB`
- 전체 모델 크기: `56.19 MB`

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/6e73cf2f-43c9-46dd-9ef1-97b3e4bf06d9/image.png" width="60%">
</p>

### 하이퍼파라미터

CelebA 학습에 사용한 핵심 설정값은 `src/configs/celeba.yml` 기준으로 아래와 같다.

#### 공통

| 항목 | 키 | 값 |
|---|---|---|
| 실행 디바이스 | `device` | `mps` |
| 시드 | `seed` | `42` |

#### 데이터

| 항목 | 키 | 값 |
|---|---|---|
| 데이터 경로 | `data.root` | `data/celeba_32x32` |
| 입력 해상도 | `data.image_size` | `32` |
| 좌우 반전 | `data.random_horizontal_flip` | `true` |
| 로더 워커 수 | `data.num_workers` | `4` |
| 데이터셋 클래스 | `data.dataset_cls` | `datasets.CelebAImageFolder` |

#### 모델

| 항목 | 키 | 값 |
|---|---|---|
| 입력 채널 | `model.in_channels` | `3` |
| 채널 폭(nf) | `model.nf` | `64` |
| 노이즈 레벨 수(K) | `model.num_scales` | `10` |
| 최대 노이즈 | `model.sigma_begin` | `1.0` |
| 최소 노이즈 | `model.sigma_end` | `0.01` |
| dilation 설정 | `model.dilations` | `[1, 2, 4, 8]` |
| $\sigma$ 스케일링 | `model.scale_by_sigma` | `true` |

#### 학습

| 항목 | 키 | 값 |
|---|---|---|
| 배치 크기 | `training.batch_size` | `64` |
| 에폭 수 | `training.epochs` | `100` |
| 학습률 | `training.lr` | `0.0002` |
| Adam betas | `training.betas` | `[0.9, 0.999]` |
| 가중치 감쇠 | `training.weight_decay` | `0.0` |
| 그래디언트 클립 | `training.grad_clip` | `1.0` |
| 자동 재개 | `training.auto_resume` | `true` |
| 저장 주기 | `training.save_every` | `1` |
| 샘플링 주기 | `training.sample_every` | `1` |

#### 샘플링(ALD)

| 항목 | 키 | 값 |
|---|---|---|
| 샘플 수 | `sampling.n_samples` | `64` |
| 레벨당 스텝 수 | `sampling.n_steps_each` | `100` |
| step lr | `sampling.step_lr` | `2.0e-05` |
| 값 클램프 | `sampling.clamp` | `true` |
| 디노이즈 단계 | `sampling.denoise` | `false` |
| 초기 분포 | `sampling.init_distribution` | `normal` |

### 손실 곡선

<p align="center">
  <img src="https://velog.velcdn.com/images/lumerico284/post/5d5dee5f-a528-4c51-ae41-ddf36486ed78/image.png" width="80%">
</p>

Running loss 곡선을 처음부터 끝까지 살펴보면 전체적으로 학습이 안정적으로 진행되었다는 느낌을 받았지만, 최종 샘플링 결과를 함께 놓고 보면 손실이 plateau에 도달했다고 해서 모델이 완전히 수렴했다고 보기는 어렵다는 점이 명확해졌다. 초반에는 손실이 약 $7000$에서 빠르게 $1000$ 근처까지 떨어지면서 모델이 큰 $\sigma$ 구간의 거친 구조를 빠르게 학습하는 전형적인 패턴을 보였고, 이후 $600$에서 $1000$ 사이의 좁은 범위에서 긴 구간 동안 진동하며 안정적인 plateau를 형성했다. 겉보기에는 학습이 잘 진행되고 최적화가 안정된 것처럼 보이지만, 이 plateau에서의 안정화가 곧바로 데이터 분포의 충분한 커버리지를 의미하는 것은 아니라는 점을 이번 실험에서 체감했다.

실제로 최종 샘플링을 보면 얼굴의 전체적인 형태나 질감은 잡혀 있지만, 특정 색감이나 헤어 텍스처 패턴이 과하게 반복되고, 표정이나 얼굴 구조의 다양성도 제한적으로 나타나 완전한 수렴과는 거리가 있었다. 즉, running loss는 일찍이 평평해졌지만, 그 값 자체가 아직 모델이 CelebA의 manifold를 충분히 학습했다고 판단할 기준이 되지는 못한다는 것을 확인한 셈이다. 이는 DSM 손실이 본질적으로 $\mathbf{x} + \sigma \mathbf{z}$ 형태의 노이즈 복원 문제에 초점을 두기 때문에, 시각적 다양성이나 분포 커버리지 같은 생성 품질과 직접적으로 일대일 대응하지 않는 특성이 반영된 결과라고 생각한다.

결국 이번 학습 곡선은 최적화가 안정적이고 수치적으로 폭주하지는 않았지만, 여전히 더 많은 iteration이나 더 세밀한 하이퍼파라미터 조정이 필요하다는 신호로 해석하는 것이 맞는 것 같다. 특히 $\sigma$ 스케줄의 범위, `n_steps_each`, `step_lr`, 모델 용량 등을 조정해야 데이터 분포의 다양한 영역을 제대로 재현할 수 있을 것으로 보인다. 다시 말해, 손실 그래프만 보면 학습이 이미 충분히 끝난 것처럼 보이지만, 실제 샘플은 그렇지 않았기 때문에 이번 실험은 학습 안정성과 데이터 수렴이 반드시 같은 타이밍에 오지 않는다는 점을 분명하게 보여주는 과정이었다.

### 결과 샘플링

| 초기 분포 | `seed=42` | `seed=10` |
|------|------|------|
| $\mathcal{U}(-1,1)\in\mathbb{R}^{N\times H\times W}$ | <p align="center"><img src="https://velog.velcdn.com/images/lumerico284/post/c6a43ef5-3986-4483-acf5-59e54c13fdd2/image.gif" width="70%"/></p> | <p align="center"><img src="https://velog.velcdn.com/images/lumerico284/post/e8b3a26b-a1c1-47d9-bc2f-17ccf88c20cb/image.gif" width="70%"/></p> |
| $\mathcal{N}(\mathbf{0},\mathbf{I})\in\mathbb{R}^{N\times H\times W}$ | <p align="center"><img src="https://velog.velcdn.com/images/lumerico284/post/0d8a3a83-474c-49f1-b06f-31397425ab49/image.gif" width="70%"/></p> | <p align="center"><img src="https://velog.velcdn.com/images/lumerico284/post/e6c4c7ef-9545-4802-98cf-fbd955bb52b8/image.gif" width="70%"/></p> |

---

## ✅ 결론

본 구현은 NCSN의 핵심 구조를 이론적 관점에서 충실히 재현하며, DSM 학습부터 ALD 샘플링에 이르는 전체 파이프라인이 논문 식과 코드 구현 간에 일관된 대응 관계를 갖도록 설계되었다. Conditional Instance Normalization을 통한 노이즈 조건화, σ-스케일 보정 방식, RefineNet 기반 백본 구조 등 주요 구성 요소가 각 노이즈 레벨에서의 score 근사를 안정적으로 수행하도록 정교하게 구현되었으며, CelebA 실험에서도 다양한 초기 분포에서 일관된 샘플 회복 과정을 보여 score-based generative modeling의 작동 원리가 자연스럽게 드러났다.

또한 학습 곡선, 샘플링 동작, 하이퍼파라미터 스케줄의 효과 등을 종합적으로 관찰함으로써 다중 노이즈 스케일을 활용한 annealing 절차가 데이터 매니폴드 복원에 어떻게 기여하는지도 확인할 수 있었다. 무엇보다 이번 프로젝트는 “동작하는 코드”를 넘어, 이론식 → 구현 → 실험적 동작의 연결을 명확히 드러내는 교육적·연구적 가치가 있는 코드베이스를 구축했다는 점에서 의미가 크며, 이는 향후 DDPM, NCSN++, Score-SDE 등 현대적 확산/스코어 모델 연구로 확장할 수 있는 견고한 기반이 될 것이다.

---

#### 📄 출처

Song, Yang, and Stefano Ermon. _"Generative Modeling by Estimating Gradients of the Data Distribution."_ Advances in Neural Information Processing Systems, vol. 32, 2019.