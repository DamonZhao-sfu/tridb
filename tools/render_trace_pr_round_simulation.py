"""Render a synthetic selection JSON as an auditable Markdown/interactive HTML report."""
import argparse
import html
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('result', type=Path)
    args = p.parse_args()
    d = json.loads(args.result.read_text())
    assert d['synthetic_only'] and d['real_agent_calls'] == d['llm_api_calls'] == 0
    w = d['selection']
    winner_source = w['sources'][0].removeprefix('pr_')
    anchor = next(s for s in d['pr_starts'] if s['source'] == winner_source)
    fields = d['schema']['fields']
    diff = [(n, anchor['config'][n], w['config'][n]) for n in fields
            if anchor['config'][n] != w['config'][n]]
    md = [
        '# TRACE 四起点 PR：单轮合成模拟', '',
        '**合成观测＋mock LLM 提案；实际执行了 GP 拟合和 PR/acquisition 计算。未执行真实 agent profiling。**', '',
        '正式实现计划：[v0.4.1](../../docs/superpowers/plans/2026-11-02-trace-joint-encoding-pr-implementation_v0.4.1.md)。', '',
        f"Seed={d['seed']}；10 个合成初始化配置；每个配置一行合成聚合指标，没有伪造 1,000 条任务收据。",
        '连续 edge bounds=[0,1] 仅为模拟 fixture；所有算子能力在本模拟中是符号能力，不代表生产实现完成。', '',
        '## 四类起点', '', '| 起点 | 计划 | 初始 qLogNEHVI |', '|---|---|---:|'
    ]
    for s in d['pr_starts']:
        md.append(f"| {s['source']} | {s['plan']} | {s['initial_acq']:.6f} |")
    md += ['', '## 最终选择', '', f"`{w['plan']}`", '',
           f"来源：`{', '.join(w['sources'])}`；qLogNEHVI={w['qlognehvi']:.9f}。这是模型评分，不是真实 HV 增量；log 分数为负不代表收益为负。", '',
           '```json', json.dumps(w['config'], ensure_ascii=False, indent=2), '```', '',
           '相对生成它的起点，变化如下：', '', '| 字段 | 起点 | 最终配置 |', '|---|---|---|']
    md += [f'| {n} | {a} | {b} |' for n,a,b in diff]
    md += ['', '## 联合编码', '',
           '21 个决策坐标（11 categorical＋9 ordinal＋1 continuous）＋8 activity masks，共 29 维。', '',
           '```text', str(w['encoded_vector']), '```', '', '| 索引 | 字段 | 配置值 | 编码值 |', '|---:|---|---|---:|']
    for i,n in enumerate(fields):
        md.append(f"| {i} | {n} | {w['config'][n]} | {w['encoded_vector'][i]:.8g} |")
    for j,n in enumerate(d['schema']['mask_fields'],21):
        md.append(f"| {j} | active:{n} | — | {w['encoded_vector'][j]:.0f} |")
    md += ['', '本次胜出配置关闭 edge filter，所以该连续坐标 inactive；这并不意味着模拟没有优化连续变量。四条 PR 路径的 edge 控制量和逐步记录保存在 selection.json。', '',
           '## 计算量与验证', '', '| 项目 | 实际值 |', '|---|---:|',
           f"| Sobol / mock LLM / recombination 原始点 | {d['channels']['sobol']} / {d['channels']['llm_mock']} / {d['channels']['recombination']} |",
           f"| PR draws / 更新步 | {d['counts']['pr_draws']} / {d['counts']['pr_steps']} |",
           f"| PR 同批重复 draws | {d['counts']['internal_duplicate_draws']} |",
           f"| 不同 acquisition 输入 | {d['unique_acquisition_inputs']} |",
           f"| 最终公共池 | {d['final_pool_size']} |",
           f"| GP 拟合 / proposal 秒 | {d['fit_seconds']:.3f} / {d['proposal_seconds']:.3f} |",
           f"| 峰值 RSS MiB | {d['peak_rss_mib']:.2f} |",
           f"| 真实 agent / LLM API 调用 | 0 / 0 |", '',
           '已检查：全部已评分向量 decode/encode 合法；四起点不同；所选配置未出现在初始化；最终点是公共池 ell 最大者；梯度及评分有限。', '',
           '这些检查不覆盖生产 ledger、并发恢复、真实 graph/operator、100-task 执行或真实质量。GP 计算期间出现 1e-8 Cholesky jitter 警告，程序成功完成；不能据合成运行推断真实 GP 校准。', '',
           '## 可复现入口', '', '```bash',
           'source /local-scratch/localhome/hza214/miniconda3/etc/profile.d/conda.sh',
           'conda activate memopt',
           '.venv-optimizer/bin/python tools/trace_pr_round_simulation_v041.py --out <新的输出目录> --seed 20261102 --deadline 60',
           '```', '', f"版本：{d['versions']}。源码 SHA256：`{d['source_sha256']}`。同目录 simulation_source.py 是实际运行源码快照。",
           '模拟采用 LOO 首步＋此前批次 EMA；初次 LOO-every-step 探索运行保留在相邻 v041 目录，不作为本报告的计数来源。']
    (args.result.parent/'report.md').write_text('\n'.join(md)+'\n')
    esc = html.escape
    start_cards = ''.join(f"<article><h3>{esc(s['source'])}</h3><p>{esc(s['plan'])}</p><p>起点 ell = {s['initial_acq']:.4f}</p><details><summary>完整配置</summary><pre>{esc(json.dumps(s['config'],ensure_ascii=False,indent=2))}</pre></details></article>" for s in d['pr_starts'])
    encoding_rows=''.join(f'<tr><td>{i}</td><td>{esc(n)}</td><td>{esc(str(w["config"][n]))}</td><td>{w["encoded_vector"][i]:.6g}</td></tr>' for i,n in enumerate(fields))
    encoded_data=json.dumps(d['ranked_pool'],ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TRACE · PR 单轮模拟</title><style>
:root{font-family:system-ui,sans-serif;color:#172632;background:#f4f6f8}body{max-width:1180px;margin:36px auto;padding:0 20px}h1{font-size:30px;margin-bottom:10px}h2{margin-top:32px;font-size:22px}.note{background:#fff2d8;border-left:5px solid #b27c18;padding:16px;line-height:1.6}.flow,.cards,.stats{display:grid;gap:12px}.flow{grid-template-columns:repeat(5,1fr);margin:22px 0}.flow>div,article,.stat{background:white;border:1px solid #d8e0e6;border-radius:8px;padding:15px}.cards{grid-template-columns:repeat(4,1fr)}.stats{grid-template-columns:repeat(4,1fr);margin:20px 0}.stat strong{display:block;font-size:27px;color:#146b70}.winner{background:#e7f4f0;border:1px solid #9bc8bc;padding:20px;border-radius:8px}.winner code{font-size:18px}table{width:100%;border-collapse:collapse;background:white}td,th{text-align:left;padding:10px;border-bottom:1px solid #e1e6eb;vertical-align:top}th{background:#e9eef3}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f3f5;padding:12px;font-size:12px}input{width:100%;box-sizing:border-box;padding:12px;border:1px solid #a9b7c3;border-radius:6px;margin:12px 0}details{margin-top:10px}summary{cursor:pointer;color:#146b70}a{color:#146b70}.scroll{overflow:auto;max-height:540px}.foot{color:#596a77;font-size:14px;line-height:1.7}@media(max-width:800px){.flow,.cards,.stats{grid-template-columns:1fr 1fr}}
</style><h1>TRACE · 联合计划＋参数 PR 选点</h1><div class="note"><strong>合成模拟，不是 ScienceWorld 实测结果。</strong><br>10 个合成观测训练三个真实 GP；四个起点运行 PR；实际计算 qLogNEHVI；LLM 提案是 fixture。真实 agent/LLM API 调用均为 0。</div>
<div class="flow"><div>① 10 个初始化<br>合成三指标</div><div>② 3 个 GP<br>冻结 acquisition</div><div>③ 三原始通道<br>＋四起点 PR</div><div>④ 合并去重<br>统一 ell 排序</div><div>⑤ 选 1 个配置<br>本次不 profiling</div></div>
<h2>四类 PR 起点</h2><div class="cards">START_CARDS</div>
<h2>本轮最终配置</h2><div class="winner"><code>WIN_PLAN</code><p>来源 WIN_SOURCE · qLogNEHVI = WIN_SCORE</p><details><summary>展开完整配置</summary><pre>WIN_JSON</pre></details></div>
<div class="stats"><div class="stat"><strong>POOL</strong>最终合法候选</div><div class="stat"><strong>UNIQUE</strong>不同 acquisition 输入</div><div class="stat"><strong>SECONDS s</strong>本机 proposal 耗时</div><div class="stat"><strong>RSS MiB</strong>峰值 RSS</div></div>
<details><summary>查看 21 个决策坐标＋8 个 activity masks</summary><p>类别 ID 不表示距离。浮点编码不改变整数参数的离散性质。edge 关闭时，其值为 inactive。</p><table><thead><tr><th>索引</th><th>字段</th><th>最终值</th><th>编码</th></tr></thead><tbody>ENCODING_ROWS</tbody></table><pre>MASK_VALUES</pre></details>
<h2>完整候选池 · 按 qLogNEHVI 排序</h2><input id="filter" placeholder="过滤来源或计划，例如 pr_llm、beam、relational"><p id="count"></p><div class="scroll"><table><thead><tr><th>排名</th><th>来源</th><th>qLogNEHVI</th><th>计划及参数</th></tr></thead><tbody id="rows"></tbody></table></div>
<p class="foot">候选分数是模型下的 log acquisition，不是真实 HV 增量。最终点关闭 edge filter，所以连续参数 inactive；其他 PR 样本仍联合优化该维度。当前结果不证明 PR 优于其他算法，也不代表这些扩展算子已接入生产。</p>
<p><a href="report.md">阅读完整报告</a> · <a href="selection.json">原始结果 JSON</a> · <a href="simulation_source.py">运行源码快照</a> · <a href="../../docs/superpowers/plans/2026-11-02-trace-joint-encoding-pr-implementation_v0.4.1.md">代码实现计划</a></p>
<script id="data" type="application/json">RESULT_DATA</script><script>
const pool=JSON.parse(document.getElementById('data').textContent);const tbody=document.getElementById('rows');function render(){const q=document.getElementById('filter').value.toLowerCase();const rows=pool.filter(r=>JSON.stringify(r).toLowerCase().includes(q));tbody.replaceChildren();for(const r of rows){const tr=document.createElement('tr');for(const value of [r.rank,r.sources.join(', '),r.qlognehvi.toFixed(6)]){const td=document.createElement('td');td.textContent=value;tr.append(td)}const td=document.createElement('td');const text=document.createElement('div');text.textContent=r.plan;td.append(text);const details=document.createElement('details');const summary=document.createElement('summary');summary.textContent='参数';const pre=document.createElement('pre');pre.textContent=JSON.stringify(r.config,null,2);details.append(summary,pre);td.append(details);tr.append(td);tbody.append(tr)}document.getElementById('count').textContent=`显示 ${rows.length} / ${pool.length} 个候选`}document.getElementById('filter').addEventListener('input',render);render();</script></html>'''
    replacements={'START_CARDS':start_cards,'WIN_PLAN':esc(w['plan']),
      'WIN_SOURCE':esc(', '.join(w['sources'])),'WIN_SCORE':f"{w['qlognehvi']:.6f}",
      'WIN_JSON':esc(json.dumps(w['config'],ensure_ascii=False,indent=2)),
      'POOL':str(d['final_pool_size']),'UNIQUE':str(d['unique_acquisition_inputs']),
      'SECONDS':f"{d['proposal_seconds']:.2f}",'RSS':f"{d['peak_rss_mib']:.1f}",
      'ENCODING_ROWS':encoding_rows,'MASK_VALUES':esc(str(dict(zip(d['schema']['mask_fields'],w['encoded_vector'][21:])))),
      'RESULT_DATA':encoded_data}
    for token,val in replacements.items():
        page=(page.replace('<strong>RSS MiB</strong>',f'<strong>{val} MiB</strong>')
              if token == 'RSS' else page.replace(token,val))
    (args.result.parent/'index.html').write_text(page)
    print(args.result.parent/'report.md')
    print(args.result.parent/'index.html')


if __name__ == '__main__':
    main()
