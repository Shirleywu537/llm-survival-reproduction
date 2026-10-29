"""Create a self-contained Chinese report with English technical terminology."""
from pathlib import Path
import base64,json,html
import numpy as np
import pandas as pd
root=Path(__file__).resolve().parent.parent
s=pd.read_csv(root/'results/summary.csv');dd=pd.read_csv(root/'results/data_diagnostics.csv')
td=pd.read_csv(root/'results/t_model_diagnostics.csv');meta=json.loads((root/'results/metadata.json').read_text())
cfg=json.loads((root/'results/config.json').read_text())
labels={'paper':'论文分布','hidden_intent_mixture':'隐藏意图混合分布'}

def table_at(methods,source='known',extra=False):
    x=s[(s.budget==1000)&s.method.isin(methods)&(s.c_source==source)].copy()
    rows=[]
    for _,r in x.iterrows():
        rows.append([labels[r.scenario],r.model,r.method,f'{100*r.coverage_mean:.2f}% ± {100*r.coverage_std:.2f} pp',f'{r.mean_lpb_mean:,.0f}',f'{r.mean_actual_calls_mean:,.0f}'])
    return '<table><thead><tr>'+''.join('<th>'+h+'</th>' for h in ['数据','T 模型','方法','覆盖率：均值 ± SD','平均下界 L','平均 min(T,C)'])+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+v+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'

def figure(name,caption):
    raw=(root/'figures'/f'{name}.png').read_bytes();enc=base64.b64encode(raw).decode()
    return f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{enc}"><figcaption>{caption}</figcaption></figure>'

parts=['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>论文复现实验报告 | LLM time-to-unsafe-sampling</title>
<style>body{font-family:system-ui,-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;color:#172435;background:#f5f7fa;line-height:1.8;margin:0}main{max-width:1180px;margin:auto;padding:44px 48px;background:white}h1{font-size:34px;line-height:1.3;margin:0 0 18px}h2{font-size:25px;border-bottom:2px solid #dde6ee;padding-bottom:8px;margin-top:44px}h3{font-size:19px;margin:26px 0 8px}p{margin:12px 0}.subtitle{color:#5a6876}.lead{background:#edf6f2;border-left:4px solid #009e73;padding:16px 22px}.note{background:#f1f5fa;padding:16px 22px}.formula{font-family:ui-monospace,monospace;overflow:auto;padding:16px;background:#f5f7fa;white-space:pre-wrap;font-size:14px}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}th,td{border-bottom:1px solid #dfe6ed;padding:10px;text-align:left}th{background:#eef3f7}figure{margin:28px 0}img{width:100%;height:auto}figcaption{font-size:14px;color:#526171}li{margin:8px 0}a{color:#00699a}code{background:#edf1f4;padding:2px 4px}.english{color:#405569;font-size:15px}.small{font-size:13px;color:#647181}@media print{body{background:white}main{padding:0;max-width:none}figure,table{break-inside:avoid}h2,h3{break-after:avoid}}@media(max-width:750px){main{padding:24px 16px}table{font-size:11px}th,td{padding:5px}}</style><main>
<p class="subtitle">计算实验 / Computational experiments · arXiv:2506.13593v2 · 2026-10-12</p>
<h1>在有限采样预算下，如何校准<br>“第一次危险输出”的时间下界？</h1>
<p><i>Calibrated Predictive Lower Bounds on Time-to-Unsafe-Sampling in LLMs</i><br>Hen Davidov, Gilad Freidkin, Shai Feldman, Yaniv Romano</p>
<div class="lead"><b>实验已完成，尚未制作 PPT。</b>实现五种方法，以神经网络和随机森林分别拟合条件分布，在论文合成分布及更难的隐藏意图混合分布上运行。主实验使用已知的删失分布 C；补充实验用独立模拟记录学习 C。结果支持“截断和优化分配能够稳定总体覆盖率”这一主要现象，但未逐点复现原图数值，也未重现原文对所有基线方法的描述。</div>
<p class="english">We reproduce the main qualitative pattern: capping and optimized allocation stabilize marginal coverage. The numerical curves differ from Figure 1, and the harder distribution exposes a large subgroup coverage gap. Estimating censoring weights can shift coverage in either direction.</p>
<h2>1. 实验回答什么问题？</h2>
<p>T 表示同一提示词第一次产生危险输出时的生成序号，支持集为 1, 2, …；C 是预先分配的最大生成次数。校准只能观察到 min(T,C) 与 C。我们预测下界 L(X)，希望总体覆盖率 P(T ≥ L(X)) 接近或高于 90%。L 大表示更有信息，但必须结合覆盖率判断；一个巨大而经常失效的下界不是好结果。对于整数 L，该事件只约束前 L−1 次未出现危险输出。</p>
<p>预算约束为 E[Σ Cᵢ | Xcal] ≤ B，记 b = B / ncal。它是期望预算，单次随机分配可以超过 B。论文训练阶段的采样成本另计。</p>
<h2>2. 实现：五种方法与两种分布学习器</h2>
<table><thead><tr><th>方法</th><th>C 的设计与下界</th><th>要解决的问题</th></tr></thead><tbody>
<tr><td>Uncalibrated</td><td>L = q̂α(X)，α = 0.1</td><td>检验直接使用模型分位数的误差</td></tr>
<tr><td>Naive</td><td>C ∼ Geom(1/b)，独立于 X</td><td>对照：逆删失概率权重可能极大</td></tr>
<tr><td>Basic</td><td>cᵢ = q̂τprior(Xᵢ)，πᵢ = min(b/cᵢ,1)，Cᵢ = cᵢ · Bern(πᵢ)</td><td>将采样长度适配到提示词</td></tr>
<tr><td>Trimmed</td><td>f̂τ = min(q̂τ,M)，M = 10b；其余同 Basic</td><td>将真实逆概率权重上界控制在 10</td></tr>
<tr><td>Optimized</td><td>使用截断分位数，并最小化平均 1/πᵢ</td><td>在同一预算内降低校准波动</td></tr>
</tbody></table>
<p>Optimized 的凸优化由 KKT 条件得到 πᵢ = min(1, a / √cᵢ)，用二分法寻找满足 Σ cᵢπᵢ = B 的 a；若全部提示词都评估仍未用满预算，则 πᵢ = 1。实现不依赖通用凸优化求解器，并用独立的数值求解器检查了目标值。</p>
<div class="formula">α̂(τ) = (1/ncal) Σ wτ(Xᵢ) · 1{q̂τ(Xᵢ) ≤ Cᵢ} · 1{T̃ᵢ &lt; q̂τ(Xᵢ)}
wτ(X) = 1 / P(C ≥ q̂τ(X) | X)
τ̂ = 最大的候选 τ，使从 0 到 τ 的所有候选误覆盖率都不超过 α
L(X) = q̂τ̂(X)，或截断后的 f̂τ̂(X)</div>
<p>分位数使用几何分布的整数约定 q̂τ = ceil(log(1−τ)/log(1−p̂))，并明确设 q̂₀ = 0。所有校准方法共享 601 个候选水平：0，加上 600 个从 10⁻⁵ 到 τprior 的对数间隔点。Naive 逐候选检查前缀条件；三个自适应方法通过加权事件排序得到同一网格上的结果。</p>
<h3>T | X 的学习</h3>
<p>每个训练提示词模拟 500 次输出，以危险输出总数 K 作为充分统计量。DNN 使用 4 个各含 32 个单元的 ReLU 隐藏层、sigmoid 输出及二元交叉熵，目标为 K/500；AdamW，学习率 10⁻⁴，权重衰减 10⁻⁵，批量 100。RF 使用 200 棵回归树，叶子最少 40 个提示词，学习 K/500。两者均将预测危险概率代入几何分布，构成完整的 T|X 模型；p̂ 截断在 [10⁻⁸, 1−10⁻⁸]。</p>
<p>两种模型只用训练集的 X 和 K，不使用真实 p、校准 T 或测试标签。DNN 在较小训练集上运行 25 个 epoch，保留论文原设置约 4,500 次优化更新；其余超参数事先固定，没有根据测试结果调参。</p>
<h3>C | X 的学习：独立补充实验</h3>
<p>论文主动设计 C，因此其分布本来已知。为了覆盖作业对两类模型学习 T|X 与 C|X 的要求，额外为每个场景和 T 模型准备 4,000 个独立 C-only 训练提示词、2,000 个验证提示词，在每档预算模拟分配记录。它们不产生危险输出标签，不加入 T 的训练或校准集。</p>
<p>Naive 的 C 模型学习几何分布参数：DNN 最小化几何负对数似然，预测相对已知预算 b 的平均长度残差；RF 学习归一化长度 (C−1)/(b−1)，再还原几何参数。自适应方法的支持点 0 和 c(X) 由分配方案确定，模型学习 π̂(X,b)=P(C&gt;0|X,b,c)。DNN 用 BCE，RF 回归是否被选择。输入为 X、log b，以及自适应方法已知的 log c。C-DNN 使用相同的四层网络、15 个 epoch、学习率 10⁻³、批量 512；C-RF 使用 100 棵树、叶子最少 60 个样本。</p>
<p>训练后的 C 模型在所有 20 次校准重复中固定。补充实验将相同的已观测 T̃、C 分别配上已知权重和估计权重，隔离权重估计的影响。π̂ 下限为 0.001。这个实验仍使用已知的分布族与设计支持点，不是任意删失分布的完全非参数估计，也不能直接继承论文“权重精确已知”的保证。</p>
<h2>3. 数据、计算规模与更难场景</h2>
<table><tr><th>项目</th><th>设置</th></tr><tr><td>提示词数</td><td>每个场景 40,000：18,000 训练 / 18,000 校准 / 4,000 测试</td></tr><tr><td>预算 b</td><td>10, 100, 500, 1,000, 2,000, 3,500, 5,000, 6,500, 8,000, 10,000</td></tr><tr><td>校准重复</td><td>20 次；固定数据划分与拟合模型；方法间使用相同随机数配对</td></tr><tr><td>目标与截断</td><td>α = 0.1，τprior = 10⁻¹/⁴ ≈ 0.56234，γ = 10，M = floor(10b)</td></tr><tr><td>主随机种子</td><td>20261012；子实验种子偏移保存于代码</td></tr><tr><td>结果记录</td><td>4,000 条主实验记录 + 3,200 条估计 C 的补充记录 = 7,200 条</td></tr></table>
<p>所有数据均为合成数据。用二项分布直接生成训练计数、用几何分布直接生成 T，与逐次模拟 Bernoulli 输出在该模型下等价，避免实际生成文本。覆盖率在 4,000 个测试 X 上用真实生存概率取平均，因此没有额外的测试 T 抽样噪声；测试 X 仍是有限样本。固定一次划分也意味着误差带不包含重新抽取数据或重新训练模型的不确定性。</p>
<h3>论文分布</h3>
<p>以 90% 概率令 log₁₀p ∼ Uniform[−4,−3]，以 10% 概率令 log₁₀p ∼ Uniform[−6,−5]。10 个特征的均值来自不同真实几何分位数的四次方根，使用训练集总体均值归一化，并加入标准差 0.1 的高斯噪声。真实 p 仅用于合成数据及最后评估。</p>
<h3>更难分布：隐藏意图 + 少数语言 + 非线性交互</h3>
<p>观测特征表示敏感主题、混淆程度、上下文长度、安全措辞、语言群体和无关嵌入坐标。少数语言标记 g ∼ Bernoulli(0.1)。未被这些摘要完整捕捉的危险意图 H 在一个提示词的生成序列内保持固定，H|X ∼ Bernoulli(w(X))。当 H=0 时风险很低，当 H=1 时风险显著增大。</p>
<div class="formula">z₀,…,z₄ ∼ Uniform[−1,1]；I = 1{z₀&gt;0.35 且 z₁&gt;0.35}
w(X) = clip(0.015 + 0.09g + 0.16I + 0.025·1{z₂&gt;0.5}, 0.005, 0.40)
log₁₀ p_low = −5.25 + 0.55z₀ + 0.30z₂ − 0.30z₃ + 0.20 sin(πz₄)
log₁₀ p_high = −3.15 + 0.45z₀ + 0.35z₁ + 0.30g + 0.30I + 0.20 sin(πz₂z₄)
p_low 截断在 [10⁻⁷, 2×10⁻⁴]；p_high 截断在 [10⁻⁴, 0.025]
k = max(L−1,0)
P(T ≥ L | X) = (1−w)(1−p_low)^k + w(1−p_high)^k</div>
<p>现实解释是：具有相似粗粒度特征的提示词可能具有不同实际意图，少数语言和“敏感主题 × 混淆表达”的组合更难被摘要特征识别。以上系数是人为设定的压力测试，不能当作真实语言群体的风险结论。</p>
<p>在完整提示词及其隐藏意图给定后，生成可以独立；仅给定观测摘要 X 后，T|X 是两个几何分布的混合，且各次输出共享 H。这有意超出论文的单一几何工作模型。每次校准重复固定观测特征 X，并从 T|X 重新抽取 H 和 T；因此重复的是同一特征分布下的生成序列。校准仍满足 C 与 T 在 X 给定后独立，因为采样分配只使用观测 X 和独立随机数。</p>
''']
paper=dd[dd.scenario=='paper'].iloc[0];hard=dd[dd.scenario=='hidden_intent_mixture'].iloc[0]
parts.append(f'<p>难度变化可量化：训练集 500 次都未出现危险输出的提示词从 <b>{paper.zero_label_fraction:.2%}</b> 增加到 <b>{hard.zero_label_fraction:.2%}</b>；潜在意图不可观测使单一几何分布无法准确表达尾部。两种模型的 Optimized 条件覆盖误差在 b=1,000 时均高于论文场景，即使总体覆盖率依然接近 90%。</p>')
parts.append('<h2>4. 结果：近似重现 Figure 1 的主要现象</h2>')
parts.append(figure('figure1_paper','图 1｜论文分布。行分别为 DNN 与 RF；列分别为覆盖率、分配的 C、平均 LPB。误差带为 20 次校准重复的 ±1 个样本标准差。LPB 使用对数纵轴。'))
parts.append('<p>Naive 在低预算明显欠覆盖；Basic 对极小的预测危险概率仍然敏感。Trimmed 与 Optimized 从 b=100 起总体覆盖率接近 90%。当 b=10 时，M=100 的截断使下界保守，覆盖率约 96.6%，并非恰好 90%。</p><p>Optimized 的真实平均逆概率权重在所有组合中均不高于 Trimmed，符合优化目标。40 个“场景 × 模型 × 预算”组合中，它的实测覆盖率 SD 在 30 个组合较小、8 个相同、2 个略大。因此本次支持总体稳定性的改善，而非每个有限重复组合都严格更优。</p><p>本次没有复现“Naive 在较大预算过覆盖”或“Basic 普遍接近 90%”这两个原图现象。概率极小的提示词缺少危险标签，导致 DNN 和 RF 分别有约 9.1% 与 3.25% 的测试预测达到 10⁻⁸ 下限；大下界和不可靠覆盖率因此同时出现。这是需要保留的实测差异，不能只展示最像论文的曲线。</p>')
parts.append(table_at(['Uncalibrated','Naive','Basic','Trimmed','Optimized']))
parts.append('<p class="small">上表均为 b=1,000、已知 C。pp 表示百分点。平均 min(T,C) 是若第一次危险输出后立即停止时的模拟调用量；Uncalibrated 不执行校准，因此该列为 0，训练开销仍然存在。</p>')
parts.append(figure('figure1_hidden_intent_mixture','图 2｜隐藏意图混合分布。截断方法总体更稳定，低预算下更保守；Naive 与 Basic 有较大波动。'))
parts.append(figure('optimized_lpb','图 3｜单独放大 Optimized 的平均下界。扩大预算能够放宽截断。DNN 与 RF 没有统一优胜者，需同时检查总体与分组覆盖率。'))
parts.append('<h2>5. 学习 C 的结果</h2>')
parts.append(figure('learned_c_comparison','图 4｜实线使用已知 C，虚线使用同模型家族估计的 C。此图展示 Naive 与 Optimized；全部方法见 CSV 和下一张图。'))
parts.append(figure('learned_c_effect','图 5｜每次配对运行的“估计权重覆盖率 − 已知权重覆盖率”。横轴采用对数尺度，纵轴单位为百分点，正负都可能发生。'))
parts.append(table_at(['Optimized'],source='learned'))
parts.append('<p>例如论文分布、DNN、b=1,000 时，Optimized 的覆盖率从已知 C 的 90.23% 降到估计 C 的 89.72%。困难分布、RF 的 Basic 则从约 89.09% 升到 94.06%，更保守而下界更小。估计权重不保证把覆盖率往目标方向修正。某些低预算组合的 SD 为零，是截断或 τprior 限制使所有重复给出相同下界，不代表总体不确定性为零。</p>')
parts.append('<h2>6. 总体校准与群体校准不同</h2>')
parts.append(figure('subgroup_coverage','图 6｜困难分布中 Optimized 的总体与分组覆盖率。DNN 的少数语言群体存在显著欠覆盖；RF 的分组表现随预算改善。'))
parts.append('<p>在 b=1,000 时，DNN 的总体覆盖率为 <b>90.37%</b>，少数语言群体为 <b>81.23%</b>；RF 分别为 <b>90.34%</b> 与 <b>86.38%</b>。这并不与总体（marginal）目标矛盾：本文方法没有给每个 X 或每个群体都达到 90% 的保证。更高平均 LPB 也可能伴随群体间不均衡。</p>')
parts.append(figure('sampling_cost','图 7｜分配上限 C 与第一次危险输出后停止的 min(T,C)。它们分别对应论文式预算统计和一种可节约开销的实际执行方式。'))
parts.append('<h2>7. 复现范围、验证和局限</h2>')
parts.append('''<table><thead><tr><th>项目</th><th>本实验的处理</th></tr></thead><tbody>
<tr><td>指定论文版本</td><td>以 arXiv v2 的公式、p 范围、τprior、γ 为主。作者公开代码是另一个参数配置，不直接当作 v2 的逐点数值基准。</td></tr>
<tr><td>公开代码差异</td><td>审阅的提交 fd78d544a62cae88da97c056b537c497d0763848 使用高一个数量级的 p 范围、τprior≈0.1778、最大权重 100；本实验使用 v2 的 p 范围、τprior≈0.5623、γ=10。</td></tr>
<tr><td>覆盖率约定</td><td>依照正文定义使用 P(T≥L)=(1−p)^(L−1)，不是 P(T&gt;L)。原文实验节概率方向存在排版不一致；这里以目标下界事件为准。</td></tr>
<tr><td>规模缩减</td><td>100,000 缩到 40,000；仍保留 20 次重复、500 个训练输出。预算在低预算区增加密度。训练只归一化训练集均值。</td></tr>
<tr><td>求解与网格</td><td>KKT + 二分法替代通用优化器；候选 τ 使用有限网格、整数几何分位数。这是近似数值实现，未形式化证明离散实现与论文连续条件完全等价。</td></tr>
<tr><td>成本统计</td><td>在全部校准提示词上统计 C（包括 C=0 的分配）；同时报告 min(T,C)。没有真实 LLM 调用或 API 账单。</td></tr>
<tr><td>不确定性</td><td>均值 ± SD 衡量固定数据、固定拟合模型下的校准随机性；不是 95% 置信区间，也不是跨数据集或跨训练种子的结论。</td></tr>
</tbody></table>''')
eps=np.sqrt((2*cfg['max_weight']**2+5)/(cfg['n_prompts']*cfg['cal_fraction'])*np.log(20))
parts.append(f'<p>论文 Theorem 5.2 给出的是带有限样本松弛项的 PAC 下界。将 ncal=18,000、γ=10、δ=0.05 代入，松弛项约为 {eps:.3f}，对应理论表达式约 {0.9-eps:.3f}，而不是无条件保证实测覆盖率恰好 90%。本实验的 90% 是名义目标；结果图展示经验表现，不构成对理论所有前提的证明。</p>')
parts.append('<p>七项数值测试覆盖整数生存概率、独立数值优化器的目标值、预算与权重上界、事件排序校准与直接计算的一致性、Naive 前缀停止规则、IPW 固定分位数无偏性以及混合分布生存概率。完整输出另外检查预算、权重、配对比较和有限数值，并用相同配置重跑检查复现性。详细检查记录见 results/validation.json。</p>')
parts.append(f'<p>正式运行耗时约 <b>{meta["total_seconds"]:.1f} 秒</b>（不含环境安装、作图及验证重跑），平台为 {html.escape(meta["platform"])}，CPU 运行。快速来自直接抽取合成分布和事件排序，不应外推为真实 LLM 审计的运行时间。</p>')
parts.append('''<h2>8. 文件与可复现性</h2>
<ul><li><b>code/</b>：完整实现、配置、七项测试、作图和报告生成脚本；requirements.txt 记录依赖版本。</li><li><b>results/runs.csv</b>：7,200 条逐次结果；summary.csv 汇总所有方法、模型、预算和 C 方案。</li><li><b>results/models/</b>：全部拟合模型和测试预测；data_diagnostics.csv、t_model_diagnostics.csv、c_model_diagnostics.csv 记录数据与模型诊断。</li><li><b>figures/</b>：七张图，每张提供 PNG、SVG、PDF；results_at_budget_1000.csv 为代表性预算表。</li><li><b>README.txt</b>：重新运行实验、测试、图表与报告的命令。</li></ul>
<h2>参考来源</h2>
<p><a href="https://arxiv.org/pdf/2506.13593v2">论文 v2：Sections 4–6、Appendices B 与 E.1</a> 提供方法、实验设计及 Figure 1。<a href="https://github.com/giladfrid009/llm-survival/tree/fd78d544a62cae88da97c056b537c497d0763848">作者代码的固定提交</a>用于核查实现选择与版本差异。本报告所有数值图均由本次独立实现的合成实验产生。</p>
<p class="small">分析范围：合成数据上的近似复现与压力测试；未运行 RealToxicityPrompts / Llama 的真实模型实验。</p>
</main></html>''')
(root/'Experiment_Report.html').write_text('\n'.join(parts),encoding='utf-8')
print(root/'Experiment_Report.html')
