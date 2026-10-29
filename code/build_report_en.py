"""Build the English companion report directly from the existing result files."""
from pathlib import Path
import base64
import html
import json
import numpy as np
import pandas as pd

root = Path(__file__).resolve().parent.parent
s = pd.read_csv(root / 'results/summary.csv')
dd = pd.read_csv(root / 'results/data_diagnostics.csv')
td = pd.read_csv(root / 'results/t_model_diagnostics.csv')
meta = json.loads((root / 'results/metadata.json').read_text())
cfg = json.loads((root / 'results/config.json').read_text())
validation = json.loads((root / 'results/validation.json').read_text())
labels = {'paper': 'Paper distribution', 'hidden_intent_mixture': 'Hidden-intent mixture'}

def result(scenario, family, method='Optimized', source='known', budget=1000):
    rows = s[(s.scenario == scenario) & (s.model == family) & (s.method == method) &
             (s.c_source == source) & (s.budget == budget)]
    assert len(rows) == 1
    return rows.iloc[0]

def pct(value):
    return f'{100*value:.2f}%'

def table_at(methods, source='known'):
    data = s[(s.budget == 1000) & s.method.isin(methods) & (s.c_source == source)]
    rows = []
    for _, r in data.iterrows():
        rows.append([labels[r.scenario], r.model, r.method,
                     f'{100*r.coverage_mean:.2f}% ± {100*r.coverage_std:.2f} pp',
                     f'{r.mean_lpb_mean:,.0f}', f'{r.mean_actual_calls_mean:,.0f}'])
    heads = ['Distribution', 'T model', 'Method', 'Coverage: mean ± SD', 'Mean LPB L', 'Mean min(T,C)']
    return '<div class="table-scroll"><table><thead><tr>' + ''.join('<th>'+h+'</th>' for h in heads) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+v+'</td>' for v in row)+'</tr>' for row in rows) + '</tbody></table></div>'

def figure(name, caption):
    image = base64.b64encode((root / 'figures' / (name + '.png')).read_bytes()).decode()
    return f'<figure><img alt="{html.escape(caption)}" src="data:image/png;base64,{image}"><figcaption>{caption}</figcaption></figure>'

paper = dd[dd.scenario == 'paper'].iloc[0]
hard = dd[dd.scenario == 'hidden_intent_mixture'].iloc[0]
pdnn = result('paper', 'DNN')
prf = result('paper', 'RF')
hdnn = result('hidden_intent_mixture', 'DNN')
hrf = result('hidden_intent_mixture', 'RF')
pdnn_learned = result('paper', 'DNN', source='learned')
hrf_basic = result('hidden_intent_mixture', 'RF', method='Basic')
hrf_basic_learned = result('hidden_intent_mixture', 'RF', method='Basic', source='learned')
opt = s[(s.c_source == 'known') & (s.method == 'Optimized')].set_index(['scenario', 'model', 'budget'])
trim = s[(s.c_source == 'known') & (s.method == 'Trimmed')].set_index(['scenario', 'model', 'budget'])
delta = opt.coverage_std - trim.coverage_std
lower = int((delta < -1e-10).sum()); higher = int((delta > 1e-10).sum()); equal = len(delta)-lower-higher
floor_dnn = td[(td.scenario == 'paper') & (td.model == 'DNN')].iloc[0].fraction_at_probability_floor
floor_rf = td[(td.scenario == 'paper') & (td.model == 'RF')].iloc[0].fraction_at_probability_floor

parts = ['''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Experimental Report — Calibrated Lower Bounds on Time-to-Unsafe-Sampling</title>
<style>
body{font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#172435;background:#f5f7fa;line-height:1.75;margin:0}
main{max-width:1180px;margin:auto;padding:44px 48px;background:white}h1{font-size:34px;line-height:1.25;margin:0 0 18px;max-width:1000px}
h2{font-size:25px;border-bottom:2px solid #dde6ee;padding-bottom:8px;margin-top:44px}h3{font-size:19px;margin:26px 0 8px}p{margin:12px 0}
.subtitle{color:#5a6876;font-size:14px}.lead{background:#edf6f2;border-left:4px solid #009e73;padding:18px 22px;margin:22px 0}
.formula{font-family:ui-monospace,monospace;overflow:auto;padding:16px;background:#f5f7fa;white-space:pre-wrap;font-size:14px;line-height:1.85}
.table-scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}th,td{border-bottom:1px solid #dfe6ed;padding:10px;text-align:left;vertical-align:top}th{background:#eef3f7}
figure{margin:30px 0}img{width:100%;height:auto}figcaption{font-size:14px;color:#526171;line-height:1.6}li{margin:8px 0}a{color:#00699a}code{background:#edf1f4;padding:2px 4px}.small{font-size:13px;color:#647181}
@media print{body{background:white}main{padding:0;max-width:none}figure{break-inside:avoid}tr{break-inside:avoid}thead{display:table-header-group}h2,h3{break-after:avoid}.table-scroll{overflow:visible}}
@media(max-width:750px){main{padding:24px 16px}h1{font-size:28px}table{font-size:12px}th,td{padding:6px}}
</style></head><body><main>
<p class="subtitle">EXPERIMENTAL REPORT · arXiv:2506.13593v2<br>Experiments completed October 12, 2026 · English report prepared October 13, 2026</p>
<h1>Calibrated Lower Bounds on Time-to-Unsafe-Sampling: Reproduction and Synthetic Stress Tests</h1>
<p>Independent experiments based on <i>Calibrated Predictive Lower Bounds on Time-to-Unsafe-Sampling in LLMs</i>, by Hen Davidov, Gilad Freidkin, Shai Feldman, and Yaniv Romano.</p>
<div class="lead"><b>Abstract.</b> We implement five prediction and calibration methods and evaluate them using deep neural networks (DNNs) and random forests (RFs). We study both the paper's synthetic distribution and a more challenging distribution with hidden unsafe intent, nonlinear interactions, and a minority-language subgroup. Primary experiments use the known censoring distribution designed by the sampling procedure. Supplementary experiments estimate that distribution from independent pilot records. Capping and optimized allocation stabilize marginal coverage, but some baseline behaviors differ from the published Figure 1. The harder distribution exposes a substantial subgroup coverage gap, and learned censoring weights can shift coverage in either direction.</div>
<p>This report uses the same completed experiments, figures, and result files as the Chinese companion report. No additional experiments or presentation slides were produced for this translation.</p>
<h2>1. Objective and evaluation target</h2>
<p>Let T be the index of the first unsafe output for a prompt, with support {1, 2, …}. Let C be the assigned maximum number of generations. Calibration observes the censored outcome T̃ = min(T,C) and the assigned limit C. The objective is to construct a lower prediction bound L(X) whose marginal coverage, P(T ≥ L(X)), is close to or above the nominal 90% level.</p>
<p>A larger L can be more informative, but it must be interpreted together with coverage. Very large bounds that frequently exceed T are not useful evidence of safety. For an integer L ≥ 1, the event T ≥ L means that the first L−1 outputs are safe; it does not assert that all of the first L outputs are safe.</p>
<div class="formula">E[Σᵢ Cᵢ | Xcal] ≤ B,     b = B / ncal</div>
<p>The budget constraint is conditional on the calibration features and holds in expectation. A realized random allocation can exceed B. The training-generation budget is separate from the calibration budget.</p>
<h2>2. Implementation: five methods and two model families</h2>
<table><thead><tr><th>Method</th><th>Allocation and prediction rule</th><th>Purpose</th></tr></thead><tbody>
<tr><td>Uncalibrated</td><td>L = q̂α(X), with α = 0.1</td><td>Evaluate direct use of fitted conditional quantiles.</td></tr>
<tr><td>Naive</td><td>C ∼ Geom(1/b), independently of X</td><td>Provide a nonadaptive baseline whose inverse-censoring weights may be very large.</td></tr>
<tr><td>Basic</td><td>cᵢ = q̂τprior(Xᵢ); πᵢ = min(b/cᵢ,1); Cᵢ = cᵢ · Bern(πᵢ)</td><td>Adapt the assigned sampling length to the prompt.</td></tr>
<tr><td>Trimmed</td><td>Replace q̂τ by f̂τ = min(q̂τ,M), with M = 10b; otherwise use Basic.</td><td>Bound the true inverse-selection weights by 10.</td></tr>
<tr><td>Optimized</td><td>Use the trimmed quantiles and minimize the mean of 1/πᵢ subject to the budget.</td><td>Improve the allocation of sampling effort and reduce calibration variability.</td></tr>
</tbody></table>
<p>For Optimized, the Karush–Kuhn–Tucker conditions yield πᵢ = min(1, a/√cᵢ). We use bisection to find a such that Σᵢ cᵢπᵢ = B. If all prompts can be fully evaluated within the budget, we set every πᵢ to 1. This implementation avoids a general-purpose convex solver; an independent numerical optimizer was used to check the objective value in a unit test.</p>
<div class="formula">α̂(τ) = (1/ncal) Σᵢ wτ(Xᵢ) · 1{q̂τ(Xᵢ) ≤ Cᵢ} · 1{T̃ᵢ &lt; q̂τ(Xᵢ)}
wτ(X) = 1 / P(C ≥ q̂τ(X) | X)
τ̂ = largest grid level for which every grid level from 0 through τ̂ has α̂(τ) ≤ α
L(X) = q̂τ̂(X), or the trimmed value f̂τ̂(X)</div>
<p>For Trimmed and Optimized, the estimator above also uses f̂τ in place of q̂τ. Because C is known even when T is censored, the two indicator conditions identify observable failures below a candidate bound.</p>
<p>We use integer geometric quantiles, q̂τ = ceil(log(1−τ)/log(1−p̂)) for τ &gt; 0, and explicitly set q̂₀ = 0. All calibration methods share 601 candidate levels: 0 and 600 logarithmically spaced values from 10⁻⁵ to τprior. Naive checks the prefix condition directly. The adaptive methods use weighted event ordering, which was checked against direct calculation on the same grid.</p>
<h3>Learning T | X</h3>
<p>For each training prompt, we simulate 500 outputs and retain the unsafe-output count K. Directly drawing K from its binomial distribution is sufficient for the chosen training losses; it avoids storing 500 individual binary labels per prompt.</p>
<p>The DNN has four ReLU hidden layers of 32 units each and a sigmoid output. It is trained on K/500 using binary cross-entropy, AdamW, learning rate 10⁻⁴, weight decay 10⁻⁵, and batch size 100. The RF contains 200 regression trees with at least 40 prompts per leaf and also learns K/500. Both families convert the fitted unsafe probability p̂(X) into a geometric working distribution for T|X. Predicted probabilities are clipped to [10⁻⁸, 1−10⁻⁸].</p>
<p>Both learners use only training X and K. True probabilities, calibration outcomes, and test labels are excluded from fitting. The DNN runs for 25 epochs on the reduced training set, retaining approximately 4,500 optimizer updates, comparable to the original paper's training setup. Hyperparameters were fixed rather than selected using test results.</p>
<h3>Learning C | X: supplementary experiments</h3>
<p>In the paper, C is deliberately designed, so its distribution is known. To additionally compare two model families for both T|X and C|X, we simulate independent censoring-only pilot records. For each scenario and fitted T model, the pilot contains 4,000 training prompts and 2,000 held-out diagnostic prompts, with allocation records simulated at each of the ten budgets. These prompts contribute no unsafe-output labels and do not enter the T training or calibration sets.</p>
<p>For Naive, we fit a geometric C distribution. The DNN minimizes geometric negative log-likelihood and predicts a residual η relative to the known budget, giving mean C = 1 + (b−1)exp(η). The RF learns the normalized length (C−1)/(b−1) and converts its prediction back to a geometric parameter. For adaptive methods, the support points 0 and c(X) remain known from the allocation design; the model learns π̂(X,b) = P(C&gt;0|X,b,c). The DNN uses binary cross-entropy, and the RF regresses the selection indicator.</p>
<p>C-model inputs are X, log b, and, for adaptive allocations, the known log c. C-DNN inputs are standardized using only the pilot training features. The C-DNN uses the same four-layer architecture, 15 epochs, learning rate 10⁻³, and batch size 512. The C-RF uses 100 trees with at least 60 observations per leaf. Estimated adaptive selection probabilities are clipped below at 0.001.</p>
<p>The fitted C models remain fixed across all 20 calibration repetitions. Within each repetition, known-weight and learned-weight calculations use exactly the same observed T̃ and C, isolating the effect of weight estimation. These experiments retain the known distribution family and design support; they are not fully nonparametric estimates of an arbitrary censoring distribution. The paper's known-weight guarantee cannot simply be carried over to this extension.</p>
<h2>3. Data, computational scale, and the harder distribution</h2>
<table><thead><tr><th>Setting</th><th>Value</th></tr></thead><tbody>
<tr><td>Prompts per scenario</td><td>40,000: 18,000 training, 18,000 calibration, and 4,000 test.</td></tr>
<tr><td>Expected budget b</td><td>10, 100, 500, 1,000, 2,000, 3,500, 5,000, 6,500, 8,000, and 10,000.</td></tr>
<tr><td>Calibration repetitions</td><td>20 with fixed features, split, and fitted models; common random numbers pair the comparisons.</td></tr>
<tr><td>Nominal error and cap</td><td>α = 0.1; τprior = 10^(−1/4) ≈ 0.56234; γ = 10; M = floor(10b).</td></tr>
<tr><td>Primary random seed</td><td>20261012; offsets for individual components are recorded in the code.</td></tr>
<tr><td>Saved result rows</td><td>4,000 primary known-C records plus 3,200 learned-C records: 7,200 in total.</td></tr>
</tbody></table>
<p>All data are synthetic. We directly simulate binomial counts and geometric times, which is distributionally equivalent to simulating the corresponding Bernoulli generation sequences. No LLM text generation or API calls are performed. Coverage is evaluated by averaging the exact true conditional survival probabilities over the 4,000 test feature vectors. This removes additional test-T Monte Carlo noise, while uncertainty from the finite set of test X remains.</p>
<p>Shaded bands show ±1 sample standard deviation across the 20 calibration draws. They do not include variability from resampling the feature dataset or retraining the models, and they are not 95% confidence intervals.</p>
<h3>Paper distribution</h3>
<p>With probability 0.9, log₁₀p is uniform on [−4,−3]; otherwise it is uniform on [−6,−5]. The means of ten features are constructed from the fourth roots of true geometric quantiles at levels equally spaced from 0.1 to 0.9. We normalize them using the overall mean computed on training examples and add independent Gaussian noise with standard deviation 0.1. True p is used only to generate the synthetic data and evaluate predictions.</p>
<h3>Harder distribution: hidden intent, a minority subgroup, and nonlinear interactions</h3>
<p>Observed features represent sensitive-topic content, obfuscation, context length, safety framing, a language-group indicator, and nuisance embedding coordinates. The minority-language indicator is g ∼ Bernoulli(0.1). Hidden unsafe intent H is not fully captured by these summaries. Conditional on X, H is Bernoulli with probability w(X), and remains fixed within one prompt's generation sequence. Risk is low when H=0 and higher when H=1.</p>
<div class="formula">z₀,…,z₄ ∼ Uniform[−1,1];   I = 1{z₀&gt;0.35 and z₁&gt;0.35}
w(X) = clip(0.015 + 0.09g + 0.16I + 0.025·1{z₂&gt;0.5}, 0.005, 0.40)
log₁₀ p_low  = −5.25 + 0.55z₀ + 0.30z₂ − 0.30z₃ + 0.20 sin(πz₄)
log₁₀ p_high = −3.15 + 0.45z₀ + 0.35z₁ + 0.30g + 0.30I + 0.20 sin(πz₂z₄)
p_low is clipped to [10⁻⁷, 2×10⁻⁴]; p_high to [10⁻⁴, 0.025]
k = max(L−1,0)
P(T ≥ L | X) = (1−w)(1−p_low)^k + w(1−p_high)^k</div>
<p>A plausible interpretation is that prompts with similar coarse features can conceal different underlying intent, and that a combination of sensitive content and obfuscated wording can carry additional risk. The subgroup and all coefficients are hypothetical stress-test choices, not findings about any real language community.</p>
<p>Generations can be independent conditional on the complete prompt and its hidden intent. Conditional only on the observed summary X, they share H, and T|X is a mixture of two geometric distributions. This deliberately misspecifies the single-geometric working model. Each calibration repetition fixes X and redraws H and T from the conditional distribution; H is not held fixed across repetitions. Censoring remains independent of T conditional on X because allocation depends only on observed features and independent randomization, with the fitted model and calibration-feature set held fixed.</p>
''']
parts.append(f'<p>The fraction of training prompts with no unsafe outputs in 500 draws increases from <b>{paper.zero_label_fraction:.2%}</b> to <b>{hard.zero_label_fraction:.2%}</b>. Hidden intent also makes the conditional tail impossible to represent exactly with a single geometric parameter. At b=1,000, the Optimized conditional-coverage RMSE, defined as the square root of the test average of (P(T≥L|X)−0.9)², increases from {pdnn.conditional_coverage_rmse_mean:.3f} to {hdnn.conditional_coverage_rmse_mean:.3f} for the DNN and from {prf.conditional_coverage_rmse_mean:.3f} to {hrf.conditional_coverage_rmse_mean:.3f} for the RF. Thus the stress test is more difficult conditionally, even though marginal coverage remains close to the target.</p>')
parts.append('<h2>4. Results: approximate reproduction of Figure 1</h2>')
parts.append(figure('figure1_paper', 'Figure 1. Paper distribution. Rows show DNN and RF; columns show coverage, mean assigned limit C, and mean lower prediction bound. Bands are ±1 sample SD across 20 calibration repetitions. The LPB axis is logarithmic.'))
parts.append('<p>Naive substantially undercovers at low budgets. Basic remains sensitive to extremely small fitted unsafe probabilities. For the paper distribution, Trimmed and Optimized achieve marginal coverage close to 90% from b=100 upward. At b=10, the cap M=100 makes the bounds conservative, with coverage around 96.6% rather than exactly 90%.</p>')
parts.append(f'<p>Optimized has no greater mean true inverse-selection weight than Trimmed in any tested combination, as expected from its optimization objective. Across the 40 scenario–model–budget combinations, its observed coverage SD is lower in {lower}, equal in {equal}, and slightly higher in {higher}. The results support an overall stability benefit, not a strict improvement in every finite set of repetitions.</p>')
parts.append(f'<p>We do not reproduce the published descriptions that Naive overcovers at larger budgets or that Basic generally stays near 90%. In the paper-distribution experiment, approximately {100*floor_dnn:.2f}% of DNN test predictions and {100*floor_rf:.2f}% of RF test predictions reach the 10⁻⁸ probability floor. Sparse failure labels and these extreme fitted probabilities are consistent with the very large LPBs and unreliable coverage of the uncapped methods. We retain these differences rather than selecting only curves that resemble the published figure.</p>')
parts.append('<h3>Representative results at b = 1,000, known C</h3>')
parts.append(table_at(['Uncalibrated', 'Naive', 'Basic', 'Trimmed', 'Optimized']))
parts.append('<p class="small">Coverage is reported as a percentage; the SD following ± is in percentage points (pp). Mean min(T,C) is the simulated generation count if sampling stops at the first unsafe output. Uncalibrated incurs no calibration calls, so this column is zero; its training cost is still present.</p>')
parts.append(figure('figure1_hidden_intent_mixture', 'Figure 2. Hidden-intent mixture. The capped methods have more stable marginal coverage and are conservative at low budgets. Naive and Basic show greater variability.'))
parts.append(figure('optimized_lpb', 'Figure 3. Expanded view of mean Optimized LPBs. Larger budgets relax the cap and allow more informative bounds. Neither model family is uniformly preferable; coverage and subgroup behavior must also be assessed.'))
parts.append('<h2>5. Effect of estimating the censoring distribution</h2>')
parts.append(figure('learned_c_comparison', 'Figure 4. Solid lines use known censoring weights; dashed lines use weights estimated by the same model family used for T. Naive and Optimized are shown here; all methods are included in the saved results and Figure 5.'))
parts.append(figure('learned_c_effect', 'Figure 5. Paired difference: coverage with learned weights minus coverage with known weights, in percentage points. Each pair uses identical observed calibration data. Bands are ±1 SD of the paired differences.'))
parts.append('<h3>Optimized at b = 1,000, learned C</h3>')
parts.append(table_at(['Optimized'], source='learned'))
parts.append(f'<p>For the paper distribution with the DNN and b=1,000, Optimized coverage shifts from {pct(pdnn.coverage_mean)} with known C to {pct(pdnn_learned.coverage_mean)} with learned C. In the harder distribution, RF Basic shifts from {pct(hrf_basic.coverage_mean)} to {pct(hrf_basic_learned.coverage_mean)}, producing more conservative coverage and smaller bounds. Estimating the weights does not necessarily move coverage toward the nominal target.</p>')
parts.append('<p>Some low-budget combinations have zero observed SD because the cap or τprior makes all repetitions return the same bound. This is a property of the fixed design and fitted model, not evidence that total uncertainty is zero.</p>')
parts.append('<h2>6. Marginal calibration does not imply subgroup calibration</h2>')
parts.append(figure('subgroup_coverage', 'Figure 6. Overall and subgroup coverage for Optimized with known C in the harder distribution. The DNN exhibits substantial undercoverage for the simulated minority-language subgroup. RF subgroup coverage improves at larger budgets.'))
parts.append(f'<p>At b=1,000, DNN Optimized achieves overall coverage of <b>{pct(hdnn.coverage_mean)}</b>, while minority-subgroup coverage is only <b>{pct(hdnn.coverage_group_1_mean)}</b>. The RF achieves <b>{pct(hrf.coverage_mean)}</b> overall and <b>{pct(hrf.coverage_group_1_mean)}</b> for the minority subgroup. These results do not contradict a marginal target: the method does not promise 90% coverage for every X or every subgroup. A larger average LPB can coexist with substantial subgroup imbalance.</p>')
parts.append(figure('sampling_cost', 'Figure 7. Assigned generation limits C versus min(T,C), the calls required under early stopping at the first unsafe output. These are distinct measures: the former matches the designed budget, while the latter describes an execution rule that can save calls.'))
parts.append('<h2>7. Reproduction scope, validation, and limitations</h2>')
parts.append('''<table><thead><tr><th>Issue</th><th>Treatment in this implementation</th></tr></thead><tbody>
<tr><td>Requested paper version</td><td>We prioritize the formulas and parameter settings in arXiv v2. The inspected public code uses a different configuration and is not treated as a pointwise numerical reference for v2.</td></tr>
<tr><td>Public-code differences</td><td>Commit fd78d544a62cae88da97c056b537c497d0763848 uses unsafe probabilities ten times larger, τprior≈0.1778, and a maximum inverse weight of 100. We use the v2 ranges, τprior≈0.5623, and γ=10.</td></tr>
<tr><td>Coverage convention</td><td>We follow the lower-bound event P(T≥L), using (1−p)^(L−1) for L≥1. This differs from P(T&gt;L). The experiment-section display in the paper also contains a probability-direction inconsistency; we follow its stated lower-bound objective.</td></tr>
<tr><td>Reduced scale</td><td>The prompt count is reduced from 100,000 to 40,000, while retaining 20 repetitions and 500 training outputs per prompt. The budget grid is denser at low budgets. Feature normalization uses training examples only.</td></tr>
<tr><td>Optimization and grid</td><td>We use KKT-based bisection, a finite candidate grid, and integer geometric quantiles. This is an approximate numerical implementation; equivalence to all continuity assumptions of the theorem has not been formally proved.</td></tr>
<tr><td>Cost accounting</td><td>Mean C is calculated over all calibration prompts, including C=0 allocations. Mean min(T,C) is also reported. These are simulated counts, not actual LLM API usage.</td></tr>
<tr><td>Uncertainty</td><td>Mean ± SD summarizes calibration randomness conditional on a fixed dataset and fitted models. It is not a confidence interval and does not establish robustness across training seeds or independently sampled datasets.</td></tr>
</tbody></table>''')
eps = np.sqrt((2*cfg['max_weight']**2+5)/(cfg['n_prompts']*cfg['cal_fraction'])*np.log(20))
parts.append(f'<p>Theorem 5.2 in the paper provides a PAC coverage bound with a finite-sample slack term. Substituting ncal=18,000, γ=10, and δ=0.05 gives a slack of approximately {eps:.3f}, so the displayed theoretical expression is about {0.9-eps:.3f}, not an unconditional guarantee of exactly 90% empirical coverage. Here, 90% is the nominal target. The figures describe empirical behavior and do not prove that all theorem assumptions hold for this discrete implementation.</p>')
parts.append('<p>Seven numerical tests passed. They check integer survival conventions; agreement with an independent numerical optimization objective; budget and weight bounds; event-order calibration against direct calculation; the Naive prefix stopping rule; fixed-quantile inverse-probability unbiasedness by simulation; and mixture survival probabilities. Additional audits check the complete output for feasible expected budgets, bounded true adaptive weights, paired censoring comparisons, and finite metrics.</p>')
assert validation['rerun_coverage_matches'] and validation['rerun_lpb_matches']
parts.append('<p>A separate execution of the final source and configuration reproduced every saved coverage and mean-LPB value exactly in this environment. This is a reproducibility check, not an additional independent training-seed experiment. Validation records are stored in <code>results/validation.json</code>.</p>')
parts.append(f'<p>The main run took approximately <b>{meta["total_seconds"]:.1f} seconds</b> on CPU, excluding environment setup, plotting, and verification reruns. The recorded platform is {html.escape(meta["platform"])}. Direct synthetic draws and event ordering explain the short runtime; it should not be extrapolated to the cost of auditing a real LLM.</p>')
parts.append('''<h2>8. Files and reproducibility</h2>
<ul>
<li><b>code/</b>: implementation, configuration, seven tests, plotting code, and report builders. The dependency versions are pinned in <code>requirements.txt</code>.</li>
<li><b>results/runs.csv</b>: all 7,200 per-repetition result rows. <code>summary.csv</code> summarizes every scenario, model, method, budget, and censoring-weight source.</li>
<li><b>results/models/</b>: fitted model files and evaluation predictions. The data, T-model, and C-model diagnostic CSV files document sparsity and prediction quality.</li>
<li><b>figures/</b>: seven figures in PNG, SVG, and PDF formats. <code>results_at_budget_1000.csv</code> contains representative-budget results.</li>
<li><b>README.txt</b>: commands for rerunning the experiments, checks, figures, and both language versions of the report.</li>
</ul>
<p>The main experimental conclusion is that capped, optimized allocation can stabilize marginal coverage under constrained sampling, even with a misspecified working model. This does not remove the need to assess subgroup coverage, the accuracy of estimated censoring weights, and the informativeness of the resulting bounds.</p>
<h2>References</h2>
<p>Davidov, H., Freidkin, G., Feldman, S., and Romano, Y. <a href="https://arxiv.org/pdf/2506.13593v2"><i>Calibrated Predictive Lower Bounds on Time-to-Unsafe-Sampling in LLMs</i>, arXiv:2506.13593v2</a>. Sections 4–6 and Appendices B and E.1 provide the methods, experimental design, and Figure 1.</p>
<p><a href="https://github.com/giladfrid009/llm-survival/tree/fd78d544a62cae88da97c056b537c497d0763848">Authors' public code, fixed commit fd78d544a62cae88da97c056b537c497d0763848</a>, was inspected to identify implementation choices and version differences. Every numerical figure in this report was generated by the independent implementation supplied with this experiment package.</p>
<p class="small">Scope: synthetic reproduction and stress testing only. We did not run the paper's RealToxicityPrompts / Llama experiments. All seven figures are embedded, so this report can be read offline.</p>
</main></body></html>''')
output = root / 'Experiment_Report_EN.html'
output.write_text('\n'.join(parts), encoding='utf-8')
print(output)
