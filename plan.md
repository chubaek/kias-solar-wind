# Generic Data-Driven vs Physics-Informed Solar-Wind Forecasting

## 1. 핵심 질문

매일 23:30 UTC에 다음 72시간의 태양풍 속도 궤적을 예측한다.

$$
\hat{\mathbf y}_{t,1:72}
=
[\hat y_{t+1\mid t},\ldots,\hat y_{t+72\mid t}]
$$

모든 모델에는 동일한 648시간 과거 관측이 주어진다.

$$
\mathbf W_t
=
[\mathbf x_{t-647},\ldots,\mathbf x_t]
$$

$$
\mathbf x_t
=
[y_t,n_t,T_t,B_t,S_t,A_t]
$$

여기서 $y_t$는 solar-wind speed, $n_t$는 density, $T_t$는 temperature, $B_t$는
magnetic-field magnitude, $S_t$는 sunspot number, $A_t$는 coronal-hole(CH) area이다.

연구의 핵심 질문은 다음과 같다.

$$
\boxed{
\text{동일한 648시간 history가 주어졌을 때, 알려진 physics alignment를 명시하면 예측이 개선되는가?}
}
$$

Physics-informed predictor는

$$
r_{t,h}=y_{t+h-648}
$$

$$
a_{t,h}=A_{t+h-96}
$$

로 정의한다.

- $r_{t,h}$: 태양 자전에 따른 27-day recurrence
- $a_{t,h}$: CH area와 태양풍 전달 지연에 따른 4-day CH predictor

두 값은 모두 $\mathbf W_t$ 안에 존재한다.

$$
[r_{t,h},a_{t,h}]=P_h(\mathbf W_t)
$$

따라서 physics-informed model은 새로운 관측을 받는 것이 아니라, 648시간 history에서 물리적으로
중요한 위치를 명시적으로 제공받는다. 이 모델은 MHD 방정식을 직접 푸는 physics-based model이
아니라, 물리적으로 알려진 lag를 사용하는 **physics-informed empirical model**이다.

## 2. 최소 비교 모델

핵심 질문에 답하기 위해 다음 6개 모델만 비교한다.

### 2.1 Climatology ($\mu$)

$$
\boxed{
\hat y^\mu_{t+h\mid t}=\hat\mu
}
$$

$\hat\mu$는 학습 구간의 평균 solar-wind speed이다. 예측 정보가 없는 최저 기준선으로 사용한다.

### 2.2 27-day persistence ($R$)

$$
\boxed{
\hat y^R_{t+h\mid t}=y_{t+h-648}
}
$$

27-day recurrence의 단독 예측력을 측정한다.

### 2.3 4-day CH-area model ($A$)

$$
\boxed{
\hat y^A_{t+h\mid t}
=
\beta_0+\beta_1A_{t+h-96}
}
$$

계수 $\beta_0,\beta_1$은 학습 구간에서만 추정한다. CH predictor의 단독 예측력을 측정한다.

### 2.4 Combined physics-informed model ($R+A$)

$$
\boxed{
\hat y^{R+A}_{t+h\mid t}
=
\gamma_0
+\gamma_1y_{t+h-648}
+\gamma_2A_{t+h-96}
}
$$

계수 $\gamma_0,\gamma_1,\gamma_2$는 학습 구간에서만 추정한다. 두 physics predictor의
상보성을 측정한다.

### 2.5 Generic data-driven model ($\mathrm{DD}$)

648시간 history를 일반적인 temporal encoder로 요약한다.

$$
\mathbf c_t=E_\eta(\mathbf W_t)
$$

$$
\boxed{
\hat y^{\mathrm{DD}}_{t+h\mid t}
=
M_\theta(\mathbf c_t,h)
}
$$

모델에는 27일 recurrence와 4일 CH lag를 명시적으로 알려주지 않는다. 모델이 raw history와
horizon에서 필요한 관계를 학습해야 한다.

### 2.6 Physics-informed hybrid model ($\mathrm{PI}$)

Generic data-driven model과 동일한 history, encoder, output head를 사용하고 두 physics
predictor만 추가한다.

$$
\boxed{
\hat y^{\mathrm{PI}}_{t+h\mid t}
=
M_\theta(\mathbf c_t,r_{t,h},a_{t,h},h)
}
$$

$\mathrm{DD}$와 $\mathrm{PI}$의 차이는 explicit physics alignment의 유무뿐이다.

$$
\boxed{
M_\theta(\mathbf c_t,h)
\quad\text{vs.}\quad
M_\theta(\mathbf c_t,r_{t,h},a_{t,h},h)
}
$$

두 모델은 같은 encoder family, output head, loss, 학습 표본과 hyperparameter budget을
사용한다.

## 3. 각 비교가 답하는 질문

### 3.1 Physics predictor의 유효성

$$
\mu\quad\text{vs.}\quad R
$$

> 27-day recurrence가 climatology보다 유효한가?

$$
\mu\quad\text{vs.}\quad A
$$

> 4-day CH predictor가 climatology보다 유효한가?

### 3.2 두 physics predictor의 상보성

$$
R,A\quad\text{vs.}\quad R+A
$$

> Recurrence와 CH area가 서로 다른 예측 정보를 제공하는가?

### 3.3 Explicit physics alignment의 가치

$$
\boxed{
\mathrm{DD}\quad\text{vs.}\quad\mathrm{PI}
}
$$

> Raw 648시간 history를 학습하는 것만으로 충분한가, 아니면 알려진 27일·4일 alignment를
> 명시해야 하는가?

Physics-informed alignment의 기여는

$$
\boxed{
\Delta_{\mathrm{PI}}(h)
=
L_{\mathrm{DD}}(h)-L_{\mathrm{PI}}(h)
}
$$

로 정의한다.

### 3.4 Physics 외 history의 가치

$$
R+A\quad\text{vs.}\quad\mathrm{PI}
$$

> 두 physics predictor 외의 density, temperature, magnetic field 및 최근 speed·CH 변화가
> 추가 정보를 제공하는가?

그 기여는

$$
\boxed{
\Delta_{\mathrm{history}}(h)
=
L_{R+A}(h)-L_{\mathrm{PI}}(h)
}
$$

로 정의한다.

## 4. Walk-forward 평가

Forecast origin은 매일

$$
t_d=d\text{일 }23{:}30
$$

으로 고정한다. 각 origin에서

$$
s_{d,h}=t_d+h\text{ hours},
\qquad h=1,\ldots,72
$$

를 예측한다.

예를 들면

$$
\begin{array}{lll}
\text{2023-12-31 23:30}
&\rightarrow&
\text{2024-01-01 00:30 ～ 2024-01-03 23:30},\\
\text{2024-01-01 23:30}
&\rightarrow&
\text{2024-01-02 00:30 ～ 2024-01-04 23:30}
\end{array}
$$

이다.

평가 단위는

$$
\boxed{
(t_d,h,y_{t_d+h},\hat y_{t_d+h\mid t_d})
}
$$

이다. 같은 target timestamp가 여러 origin에서 예측되어도 horizon이 다르므로 모두 유지한다.
예를 들어 `2024-01-02 00:30`은 첫 origin의 $h=25$ 예측이면서 두 번째 origin의 $h=1$
예측이며, 두 forecast case를 모두 평가한다.

Private에서 매일 새로 관측된 자료로 $\mathbf W_t$만 갱신한다. 모델 파라미터는 public 학습 후
고정한다.

$$
\boxed{
\text{Private에서는 입력만 갱신하고 모델을 재학습하지 않는다.}
}
$$

Public 2011–2023에서는 같은 일별 23:30 origin과 72시간 target을 사용하여 expanding-window
chronological validation을 수행한다. 모델과 hyperparameter를 확정한 뒤 public 전체로 재학습하고,
private 2024–2025는 한 번만 평가한다.

Lookback 648시간이 완전히 존재하고 target 전체가 해당 split에 포함되는 origin만 사용한다.

$$
t-647\ge t_{\mathrm{data\ start}},
\qquad
t+72\le t_{\mathrm{split\ end}}
$$

## 5. 평가 지표와 결론 규칙

### 5.1 Horizon별 지표

Primary metric은 horizon별 MAE이다.

$$
\operatorname{MAE}_m(h)
=
\frac1{N_h}
\sum_d
\left|y_{t_d+h}-\hat y^{(m)}_{t_d+h\mid t_d}\right|
$$

보조 지표는 다음과 같다.

$$
\operatorname{RMSE}_m(h)
=
\sqrt{
\frac1{N_h}
\sum_d
\left(y_{t_d+h}-\hat y^{(m)}_{t_d+h\mid t_d}\right)^2
}
$$

$$
\operatorname{Bias}_m(h)
=
\frac1{N_h}
\sum_d
\left(\hat y^{(m)}_{t_d+h\mid t_d}-y_{t_d+h}\right)
$$

각 horizon의 correlation과 baseline 대비 skill도 함께 보고한다.

### 5.2 전체 점수

$$
\operatorname{MAE}_{m,\mathrm{all}}
=
\frac{
\sum_d\sum_{h=1}^{72}
\left|y_{t_d+h}-\hat y^{(m)}_{t_d+h\mid t_d}\right|
}{
\sum_hN_h
}
$$

인접 origin의 target 구간이 중첩되므로 paired 성능 차이의 신뢰구간은 독립 표본 bootstrap이
아니라 27일 단위 block bootstrap으로 계산한다.

### 5.3 결론 규칙

$$
L_{R+A}<\min(L_R,L_A)
$$

이면 두 physics predictor가 상보적이다.

$$
L_{\mathrm{PI}}<L_{\mathrm{DD}}
$$

이면 동일한 raw history에서도 explicit physics alignment가 유용하다.

$$
L_{\mathrm{PI}}\approx L_{\mathrm{DD}}
$$

이면 generic data-driven model이 알려진 alignment를 스스로 학습한 것이다.

$$
L_{\mathrm{PI}}>L_{\mathrm{DD}}
$$

이면 explicit physics가 중복되거나 부적절한 inductive bias를 제공한 것이다.

$$
L_{\mathrm{PI}}<L_{R+A}
$$

이면 physics predictor 외의 648시간 history가 추가 정보를 제공한다.

`더 좋다`는 결론은 전체 MAE뿐 아니라 horizon별 차이, chronological fold의 일관성 및 block
bootstrap 신뢰구간을 함께 만족할 때만 내린다.

## 6. 검증 항목

- 각 $\mathbf W_t$가 정확히 648개의 hourly observation을 포함하는지 확인한다.
- $r_{t,h}$와 $a_{t,h}$가 $\mathbf W_t$ 안의 값과 정확히 일치하는지 확인한다.
- `2023-12-31 23:30`, $h=1$이 `2024-01-01 00:30`에 연결되는지 확인한다.
- `2023-12-31 23:30`, $h=72$가 `2024-01-03 23:30`에 연결되는지 확인한다.
- 겹치는 target이 서로 다른 `(origin, horizon)` forecast case로 유지되는지 확인한다.
- 모든 모델이 동일한 origin과 target-valid mask를 사용하는지 확인한다.
- Predictor 결측 처리가 causal 또는 train-only 방식인지 확인한다.
- Private 평가 중 model parameter가 변경되지 않는지 확인한다.
- $\mathrm{DD}$와 $\mathrm{PI}$가 physics feature 외에는 동일한 architecture와 학습 조건을
  사용하는지 확인한다.

## 7. 범위와 산출물

포함하는 모델은

$$
\boxed{
\mu,\ R,\ A,\ R+A,\ \mathrm{DD},\ \mathrm{PI}
}
$$

이다.

다음 항목은 본 실험에서 제외한다.

- Handcrafted vs learned context 비교
- CH alignment 추가 ablation
- Lookback 비교
- MIMO·recursive·state-space 구조 비교
- Uncertainty quantification
- Model ensemble

최종 산출물은 다음과 같다.

- 6개 모델의 horizon별 MAE, RMSE, Bias, correlation 및 skill
- $\mathrm{DD}$ vs $\mathrm{PI}$ paired comparison
- $\Delta_{\mathrm{PI}}(h)$와 $\Delta_{\mathrm{history}}(h)$ 곡선
- Public chronological validation과 private 최종 평가를 분리한 결과표
- 실험 질문에 대한 결론과 한계
