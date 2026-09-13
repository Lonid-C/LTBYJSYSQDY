import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const require = createRequire(import.meta.url);

// marked：优先使用随附的本地解析器路径，其次回退到受管 workspace
let marked;
for (const p of [path.join(root, 'qa/marked.cjs'),
                 '/Users/Zhuanz/.workbuddy/binaries/node/workspace/node_modules/marked']) {
  try { ({marked} = require(p)); if (marked) break; } catch (e) { /* try next */ }
}
if (!marked) throw new Error('找不到 marked，请先安装：npm install marked');

const katex = require(path.join(root, 'qa/katex.cjs'));
let count = 0;
const errors = [];

function render(md) {
  const math = [];
  md = md.replace(/\$\$([\s\S]*?)\$\$/g, (_, s) => { const id = math.length; math.push([s, true]); return '\n\nQTHREEMATH' + id + 'END\n\n'; });
  md = md.replace(/\$([^$\n]+)\$/g, (_, s) => { const id = math.length; math.push([s, false]); return 'QTHREEMATH' + id + 'END'; });
  let html = marked.parse(md);
  html = html.replace(/QTHREEMATH(\d+)END/g, (_, i) => {
    const [s, displayMode] = math[+i];
    count++;
    try { return katex.renderToString(s, {displayMode, throwOnError: true, strict: false, output: 'htmlAndMathml'}); }
    catch (e) { errors.push({source: s, error: e.message}); return '<code>' + s.replaceAll('<', '&lt;') + '</code>'; }
  });
  return html.replaceAll('<table>', '<div class="table-wrap"><table>').replaceAll('</table>', '</table></div>');
}

const results = await fs.readFile(path.join(root, 'reports/RESULTS_REPORT.md'), 'utf8');
const derivation = await fs.readFile(path.join(root, 'reports/ANALYSIS_MODELING_REPORT.md'), 'utf8');
const versions = await fs.readFile(path.join(root, 'reports/版本迭代说明.md'), 'utf8');
const methods = await fs.readFile(path.join(root, 'reports/方法对比报告.md'), 'utf8');
const images = [];
const METHOD_IMAGES = [];
const FIGURES = [
  ['口径与终端条件对比', '结算口径、终端条件、优化视野与选参规则的结构对比'],
  ['策略费用分解', '各策略总费用，以及主结构变体的费用分解（减购净额为负）'],
  ['风险分位校准曲面', '评价期前窗口内风险参数全网格的库存校正费用与最低成本 0.2% 邻域'],
  ['参数邻域稳定性', '每个风险参数组合下"全更新 − 不更新"的节约，以及仍然优于不更新的组合比例'],
  ['预报更新价值', '各发布时刻的条件边际价值，以及分月节约'],
  ['参数敏感性', '单因素扰动后的真实费用变化'],
  ['指定日期调度', '四个指定日期的购电、紧急缺口与实际 SOC'],
];
const METHOD_FIGURES = [
  ['方法对比_L1最优性', '日前计划单阶段：各方法相对 LP 的间隙，以及 DP 的离散化误差收敛'],
  ['方法对比_L2全年费用', '全年滚动策略的总费用与库存校正后可比费用'],
  ['方法对比_L3多目标', '成本—风险（CVaR 与最差单日）与成本—电池寿命两条 Pareto 前沿'],
];
for (const [name, caption] of FIGURES) {
  try {
    const data = await fs.readFile(path.join(root, 'figures', name + '.png'));
    images.push(`<figure><img src="data:image/png;base64,${data.toString('base64')}" alt="${caption}"><figcaption>${caption}</figcaption></figure>`);
  } catch (e) { console.warn('missing figure', name); }
}
for (const [name, caption] of METHOD_FIGURES) {
  try {
    const data = await fs.readFile(path.join(root, 'figures', name + '.png'));
    METHOD_IMAGES.push(`<figure><img src="data:image/png;base64,${data.toString('base64')}" alt="${caption}"><figcaption>${caption}</figcaption></figure>`);
  } catch (e) { console.warn('missing method figure', name); }
}
const css = await fs.readFile(path.join(root, 'qa/katex_embedded.css'), 'utf8');

const output = `<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>第三问 · 量化推导与结果（改进版）</title><style>${css}
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:#1c2836;font:16px/1.85 -apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif}
nav{position:sticky;top:0;background:#18324d;color:white;padding:14px 24px;z-index:10;display:flex;gap:28px;flex-wrap:wrap}
nav a{color:white;text-decoration:none;font-size:14px}
main{max-width:1100px;margin:32px auto;background:white;padding:40px 52px;box-shadow:0 2px 18px #182f4710}
h1{font-size:29px;line-height:1.4;margin:0 0 28px}h2{font-size:23px;margin-top:42px;padding-bottom:9px;border-bottom:2px solid #dde5ed;scroll-margin-top:90px}
h3{font-size:19px;margin-top:30px}p{margin:13px 0}a{color:#285f94}strong{color:#183f69}
table{border-collapse:collapse;font-size:13px;line-height:1.65;width:100%}
th{background:#eaf0f6;color:#234564;text-align:left}th,td{padding:8px 12px;border-bottom:1px solid #e1e7ed;vertical-align:top}
td{font-variant-numeric:tabular-nums}tbody tr:nth-child(even){background:#f9fbfd}
.table-wrap{overflow-x:auto;margin:22px 0}.katex-display{overflow-x:auto;overflow-y:hidden;padding:10px 0}.katex{font-size:1.08em}
code{font-size:13px;background:#edf1f6;padding:2px 5px;border-radius:3px}
section{scroll-margin-top:90px}blockquote{margin:18px 0;padding:12px 18px;background:#f4f8fc;border-left:4px solid #325b88;color:#2c4258}
figure{margin:28px 0 44px}figure img{width:100%;height:auto}figcaption{font-size:14px;color:#607083;text-align:center}
li{margin:9px 0}.label{font-size:13px;color:#667d93;letter-spacing:2px;text-transform:uppercase}
.divider{border-top:4px solid #325b88;margin-top:64px;padding-top:32px}
@media(max-width:700px){main{margin:0;padding:25px 18px}body{font-size:15px}h1{font-size:24px}nav{gap:16px}}
@media print{nav{display:none}body{background:white}main{padding:0;box-shadow:none;max-width:none}.table-wrap{overflow:visible}figure{break-inside:avoid}h2,h3{break-after:avoid}.katex-display{overflow:visible}}
</style></head><body>
<nav><a href="#versions">版本迭代</a><a href="#methods">方法对比</a><a href="#results">计算结果</a><a href="#figures">图表</a><a href="#derivation">模型推导与参数来源</a><a href="result3.xlsx">结果表</a></nav>
<main><div class="label">2026 C题 · 问题三 · 改进版</div>
<section id="versions">${render(versions)}</section>
<section id="methods" class="divider">${render(methods)}<h1>方法对比图表</h1>${METHOD_IMAGES.join('')}</section>
<section id="results" class="divider">${render(results)}</section>
<section id="figures" class="divider"><h1>计算图表</h1>${images.join('')}</section>
<section id="derivation" class="divider">${render(derivation)}</section>
</main></body></html>`;

await fs.writeFile(path.join(root, '第三问_推导与结果.html'), output);
await fs.writeFile(path.join(root, 'qa/math_render_checks.json'), JSON.stringify({formula_count: count, errors}, null, 2));
if (errors.length) throw new Error(JSON.stringify(errors));
console.log(`Built offline report with ${count} rendered formulas.`);
