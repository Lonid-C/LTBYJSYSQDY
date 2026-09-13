# -*- coding: utf-8 -*-
"""由《第四问_论文正文.md》重新生成同目录下的 HTML 阅读稿。

复核意见 §10 要求"报告、Excel、CSV、JSON、图表使用同一主策略和同一版数字"。
此前 HTML 是手工维护的，与 Markdown 正文容易脱节（旧稿中 4 个图引用指向了不存在的
文件、被静默替换成了另一张图）。本脚本把 HTML 变成 Markdown 的确定性产物：
改正文只需改 .md，再跑一次本脚本。

支持的 Markdown 子集与正文用法一致：# / ## / ### 标题、段落、表格、无序/有序列表、
引用块、`$$...$$` 独立公式、`$...$` 行内公式（交给 MathJax 渲染）、**粗体**、`代码`、
以及本项目自定义的插图标记 `[[FIG:文件名|图 4-N　图题]]`。
"""
from __future__ import annotations
import html as _html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / '论文交付/2_论文部分/第四问_论文正文.md'
DST = ROOT / '论文交付/2_论文部分/第四问_论文正文.html'
FIGDIR = ROOT / '论文交付/2_论文部分/figures'
HEAD = Path(__file__).with_name('_paper_head.html')
TAIL = Path(__file__).with_name('_paper_tail.html')

MATH = []


def stash_math(t: str) -> str:
    """把 $...$ 原样挪走，避免其中的下划线、星号被当作 Markdown 语法。"""
    def keep(m):
        MATH.append(m.group(0))
        return f'\x00M{len(MATH)-1}\x00'
    return re.sub(r'\$[^$\n]+\$', keep, t)


def pop_math(t: str) -> str:
    return re.sub(r'\x00M(\d+)\x00', lambda m: MATH[int(m.group(1))], t)


def inline(t: str) -> str:
    t = stash_math(t)
    t = _html.escape(t, quote=False)
    t = re.sub(r'`([^`]+)`', lambda m: '<code>' + m.group(1) + '</code>', t)
    t = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', t)
    t = t.replace('\\_', '_')
    return pop_math(t)


def build(md: str):
    lines = md.split('\n')
    out, toc, figs = [], [], []
    i, sec, fig_alt = 0, 0, {}
    while i < len(lines):
        ln = lines[i]

        m = re.match(r'^(#{1,3}) (.+)$', ln)
        if m:
            lvl, txt = len(m.group(1)), m.group(2).strip()
            sid = f's{sec}'; sec += 1
            toc.append((lvl, sid, txt))
            out.append(f'<h{lvl} id="{sid}"><a class="hlink" href="#{sid}" '
                       f'aria-label="本节链接">§</a>{inline(txt)}</h{lvl}>')
            i += 1; continue

        m = re.match(r'^\[\[FIG:([^|]+)\|(.+?)\]\]$', ln.strip())
        if m:
            src, cap = m.group(1).strip(), m.group(2).strip()
            n = re.match(r'^(图 [\d\-]+)[　 ]*(.*)$', cap)
            num, title = (n.group(1), n.group(2)) if n else ('', cap)
            if not (FIGDIR / src).exists():
                print(f'!! 插图缺失：{src}', file=sys.stderr); sys.exit(1)
            figs.append(src)
            fig_alt[src] = title
            out.append(f'<figure class="fig"><img src="figures/{src}" '
                       f'alt="{_html.escape(title, quote=True)}" loading="lazy">'
                       f'<figcaption><span class="fig-n">{num}</span>{inline(title)}'
                       f'<a class="fig-src" href="figures/{src}" target="_blank" '
                       f'rel="noopener">查看原图</a></figcaption></figure>')
            i += 1; continue

        if ln.strip() == '$$':                       # 独立公式块
            j = i + 1
            while j < len(lines) and lines[j].strip() != '$$':
                j += 1
            body = '\n'.join(lines[i + 1:j])
            out.append('<div class="eq">$$\n' + _html.escape(body, quote=False) + '\n$$</div>')
            i = j + 1; continue

        if ln.startswith('|') and i + 1 < len(lines) and re.match(r'^\|[\s:\-|]+\|$', lines[i + 1]):
            head = [c.strip() for c in ln.strip('|').split('|')]
            align = [c.strip() for c in lines[i + 1].strip('|').split('|')]
            j = i + 2; body = []
            while j < len(lines) and lines[j].startswith('|'):
                body.append([c.strip() for c in lines[j].strip('|').split('|')]); j += 1

            def sty(a):
                if a.endswith(':') and a.startswith(':'): return ' style="text-align:center"'
                if a.endswith(':'): return ' style="text-align:right"'
                return ''
            t = ['<div class="tw"><table>', '<thead>', '<tr>']
            t += [f'<th{sty(align[k])}>{inline(c)}</th>' for k, c in enumerate(head)]
            t += ['</tr>', '</thead>', '<tbody>']
            for r in body:
                t.append('<tr>')
                t += [f'<td{sty(align[k]) if k < len(align) else ""}>{inline(c)}</td>'
                      for k, c in enumerate(r)]
                t.append('</tr>')
            t += ['</tbody>', '</table></div>']
            out.append('\n'.join(t)); i = j; continue

        m = re.match(r'^(\s*)([-*]|\d+\.) +(.*)$', ln)
        if m:                                        # 列表（含二级缩进）
            tag = 'ul' if m.group(2) in '-*' else 'ol'
            items, j, cur, depth = [], i, None, 0
            while j < len(lines):
                mm = re.match(r'^(\s*)([-*]|\d+\.) +(.*)$', lines[j])
                if mm:
                    d = 1 if len(mm.group(1)) >= 2 else 0
                    if cur is not None and d > depth:
                        items[-1][1].append(mm.group(3))
                    else:
                        items.append([mm.group(3), []])
                    cur, depth = mm, d
                    j += 1
                elif lines[j].startswith('   ') and lines[j].strip() and items:
                    items[-1][0] += ' ' + lines[j].strip(); j += 1
                elif lines[j].strip() == '' and j + 1 < len(lines) and \
                        re.match(r'^\s*([-*]|\d+\.) +', lines[j + 1] or ''):
                    j += 1
                else:
                    break
            t = [f'<{tag}>']
            for txt, sub in items:
                t.append('<li>' + inline(txt) +
                         ('<ul>' + ''.join(f'<li>{inline(x)}</li>' for x in sub) + '</ul>'
                          if sub else '') + '</li>')
            t.append(f'</{tag}>')
            out.append('\n'.join(t)); i = j; continue

        if ln.startswith('> '):
            j = i; buf = []
            while j < len(lines) and lines[j].startswith('> '):
                buf.append(lines[j][2:]); j += 1
            out.append('<blockquote><p>' + inline(' '.join(buf)) + '</p></blockquote>')
            i = j; continue

        if ln.strip() == '---':
            out.append('<hr>'); i += 1; continue

        if ln.strip():
            j = i; buf = []
            while j < len(lines) and lines[j].strip() and not lines[j].startswith(('#', '|', '>', '[[FIG')) \
                    and lines[j].strip() != '$$' and not re.match(r'^\s*([-*]|\d+\.) +', lines[j]):
                buf.append(lines[j].strip()); j += 1
            out.append('<p>' + inline(' '.join(buf)) + '</p>')
            i = j; continue
        i += 1
    return out, toc, figs


def main():
    md = SRC.read_text(encoding='utf8')
    body, toc, figs = build(md)
    tl = []
    for lvl, sid, txt in toc:
        tl.append(f'<a class="t{lvl}" href="#{sid}">{_html.escape(txt, quote=False)}</a>')
    vid = json.loads((ROOT / 'results4/fingerprint_manifest.json')
                     .read_text(encoding='utf8'))['variant_id']
    seen, uniq = set(), []
    for f in figs:
        if f not in seen:
            seen.add(f); uniq.append(f)
    figlist = '\n'.join(
        f'  <li><a href="figures/{f}" target="_blank" rel="noopener">{f[:-4]}</a></li>'
        for f in uniq)
    head = HEAD.read_text(encoding='utf8')
    tail = TAIL.read_text(encoding='utf8')
    doc = f'''{head}

<div class="wrap">
<header class="top">
  <div class="eyebrow">CUMCM C 题 · 第四问 · 交付论文手的工作稿</div>
  <h1 class="doc">波动电价下的微电网购电与储能调度</h1>
  <p class="sub">两阶段联合情景随机规划与分层融合求解。本稿为可直接改写的论文正文底稿——结构、推导、图表位置与数字均已就位。</p>
  <div class="handoff">
    <div><b>给论文手：</b>正文里的<b>每一个数字都已核验并锁定</b>，来自交付包 <code>results4/</code>，改写时请勿自行调整数值或补算。文字、语气、章节顺序都可以改；表格与图的位置可以挪。<b>但 4.6.5「模型解释边界与稳健性说明」的六条不能删</b>——参数标定与泛化边界、复杂策略的适用边界、Q4-2 的补救决策结构、信息价值指标的命名边界、结算口径敏感性、因果性与统计检验边界。这些是本稿相对其他队伍的加分项，删掉就白做了。</div>
  </div>
  <p class="sub">本页由 <code>code/q4_build_paper_html.py</code> 从 <code>第四问_论文正文.md</code> 自动生成，请勿直接改 HTML。</p>
</header>

<dl class="conv">
  <div><dt>结算口径</dt><dd>交付时段电价<br>日末净额退款</dd></div>
  <div><dt>惩罚倍率</dt><dd>增购 1.5×<br>违约 0.5×　应急 5×</dd></div>
  <div><dt>评价期</dt><dd>2025-02-01 → 12-31<br>334 天 / 48,096 段</dd></div>
  <div><dt>选参窗口</dt><dd>仅 2025 年 1 月<br>三块滚动验证</dd></div>
  <div><dt>储能</dt><dd>12,000 kWh / 5,000 kW<br>η = 0.9</dd></div>
  <div><dt>版本</dt><dd>{vid}</dd></div>
</dl>

<div class="cols">
  <nav class="toc"><div class="tl">目录</div>
{chr(10).join(tl)}</nav>
  <main>
{chr(10).join(body)}
  </main>
</div>

<footer>
  <h4>本稿引用的图（共 {len(uniq)} 张，点击查看原图；交付包内另附同名 PDF 矢量版）</h4>
  <ul>
{figlist}</ul>
  <p>全部数字与图表由交付包 <code>results4/</code> 自动生成，版本 <span class="vid">{vid}</span>；
  该版本的代码、数据、结果与交付物指纹见 <code>results4/fingerprint_manifest.json</code>，
  可用 <code>python code/q4_freeze.py</code> 复核。</p>
</footer>
</div>

{tail}'''
    DST.write_text(doc, encoding='utf8')
    print(f'✓ 生成 {DST.name}：{len(toc)} 个标题、{len(uniq)} 张图、{len(doc):,} 字节')


if __name__ == '__main__':
    main()
