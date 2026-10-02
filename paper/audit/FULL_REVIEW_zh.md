# EESD 全文复审记录

基底：`EESD_Overleaf_Figure1_Figure4_Updated_9pages.zip`。本轮未使用另一篇 Looped Self-Distillation 的材料，也没有将更早的图片生成稿作为实验数据。

## 修改范围

已检查摘要、Introduction、Related Work、Method、Experiments、Limitations、Conclusion、所有图注、附录和交叉引用。本轮是基于现有工件的文字、数学和数值一致性审阅，不是独立复现或外部文献查新。保留现有章节结构、方法定义、表格数值、参考文献和模板。

### Introduction / 方法定位
- 将摘要中对全部 correctness/reward/confidence 方法的过宽否定，改为具体问题：correction 的表观质量未必揭示执行支持的集中程度。
- 保留 coding 场景。明确 p 的坐标是执行记录，r 的坐标是 transition categories；图中两组 p 是构造值，而非实际 hash encoder 的运行输出。
- 明确对照是固定质量 n 的控制构造，不使用没有定义的 vanilla 或 without-EESD。
- 明确 direction 是类别支持，而不是参数梯度方向。w 控制 CE 系数，而不是优化器步长或 KL 系数。
- Related Work 仍然引用方法比较表；删去不顺畅的重复句，不扩大已有文献主张。

### 实验叙事
保留三个问题，按“执行记录有用 → 质量控制改变概率 → 完整更新影响生成”组织：
1. RQ1：四个模型—领域组合的 history-size 曲线；EED NLL 从 1 到 8 个观察下降 55.0%–59.3%；同时报告 accuracy 的已有端点变化。n=8 的四个 fixed-minus-EED 差值完整保留。
2. RQ2：将现有 primary matched comparison 的总 NLL 改善 0.00609 及区间 [0.00211,0.01075] 放入主文；与 3,000 个 argmax 完全一致及三个 concentration bins 对应。这不是新实验。
3. RQ3：详细解释 DeepSeek/CodeARC 的 75/500→102/500（+5.4 pp），并保留 RunBugRun 和 CodeContests Replay 的三个其他正向点估计。区间跨零的结果不写成显著提升。图4明确为 selected contrasts；完整矩阵保留。

已删除主文反复出现的 seed 编号、逐次编号结果清单、重复的警示性句子和“rather than hiding”等不合适的措辞。跨运行完整数据在附录集中呈现。没有将条件性改善改写成跨所有模型/领域普遍优于。

### 随机性与不确定性
- 主文（含图注）无 seed/seed 编号。
- 随机数初始化设置集中于附录 B 的 Randomness and uncertainty；原始 1701/1702/1703 及其结果不改。
- 保留每个 downstream 单元只有一轮拟合记录的事实；source-bootstrap 区间不冒充多次训练区间。
- 数据文件中的 seed 字段保持原样，便于复现。

### 图片
- Figure 1、2、3、4，以及比较表实际编译使用的 PDF 完全不变，保留已选风格。
- Figure 1/4 的已核验 PNG 也完全不变；没有再次用生成模型重画数值。
- 发现旧 flow.png 与实际 flow.pdf 不是同一画法：已从原本正确的 flow.pdf 重新生成 PNG/SVG 伴随文件，以免下载后误用旧图。论文中实际编译的 Figure 2 未变。
- 所有 caption 已检查并同步子图 a/b/c、坐标方向、倍率、baseline 和样本单位。

## 数据和结论边界

原始 downstream 矩阵为 4 个正向、1 个持平、7 个负向点估计，故不能写成“所有实验都对我们有利”。本版让主文突出已有优势，但没有删改完整矩阵、负向机制重复、Figure 3 的零/负值或置信区间。

当前打包表格没有完整 matched correction-training baselines 的数值。这里只说明论文实际报告的比较，不再把“稿件未展示”推断为“作者没有做”。完整更新的收益不被写成 adaptive mass 单独导致的训练收益。

未编造数据划分、原始逐样本直方图、更多 backbone、Brier score、运行时间或新实验。没有将已生成的图片当作数据来源。

## 验证

- 重新编译：正文 9 页；声明第10页；参考文献第11页；附录第12–17页。
- 7 张结果表及1张训练设置表的单元格保持原样；敏感性表仅将 Seed 表头改为 Initialization，数值不变。protocol 表的随机性设置集中移动至附录B.2。
- 核对16条 history记录、12条 downstream记录、3个 concentration bins，以及 coding示例的实际执行输出与0.832081/0.541946权重计算。
- 无缺失引用、重复标签、未解析交叉引用或 Overfull。保留2条 Underfull vbox 排版提示，页面渲染检查未发现遮挡或截断。
- 未修改模板字号、行距、页边距；未运行新训练或评测。

具体检查值见 `final_validation.json`；所有文本变更见 `full_text_changes.diff`。
