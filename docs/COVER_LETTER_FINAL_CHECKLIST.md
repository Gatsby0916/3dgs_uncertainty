# ✅ Cover Letter 最终验证清单

## Summary Review （3个要求）
- ✅ 1. FisherRF / EIG理论区别 → Sec. 1 (end), "Distinction from Expected Information Gain"
- ✅ 2. 性能收益非来自语义 → Sec. 4.1 fairness + Table 2 intrinsic + Fig.7-8 robustness
- ✅ 3. 清晰性问题 → 多处修复 (epistemic/aleatoric, 定义, 指标, 全景评估)

---

## Reviewer 1 (R1) - 评分 5/5
- ✅ R1-1: 概念新颖性 / "FisherRF + Mask?" → Sec. 3, Eq.(8)-(10) 对比
- ✅ R1-2: 运行时/内存/可扩展性 → Sec. 4.5, O(d) storage, 256× 像素减少
- ✅ R1-3: 掩码依赖和鲁棒性 → Sec. 4.4, Fig.7-8, Table 3
- ✅ R1-4: 图表密集/标题 → Fig. 4, 6 caption修改
- ✅ **R1-5: POp-GS/UNG-GS未比较** → **新增**：Sec. 2.2 和 Sec. 4.1 "Baselines" 中已提及，cover letter指向这些位置

---

## Reviewer 2 (R2) - 评分 7/7
- ✅ **R2-1: Epistemic vs Aleatoric + Exploitation regime** → Introduction, exploitation paragraph
- ✅ **R2-1b: 应用示例补充** → **新增**：Introduction, 图2下方，"cultural heritage preservation, industrial inspection, and AR"
- ✅ R2-2: 掩码如何获取未见视图 → Sec. 5, "cross-view mask prediction...future work"

---

## Reviewer 3 (R3) - 评分 6/6
- ✅ R3-1: 杂乱场景鲁棒性 → Sec. 4.4, Ghost perturbation, Fig. 8
- ✅ R3-2: 可扩展性和放宽对角FIM → Sec. 4.5 + Sec. 5 future work
- ✅ R3-3: 运行时相对GauSS-MI → Sec. 4.5, diagonal O(d) + patch-based
- ✅ R3-4: 多对象扩展 → Eq. (10) + Appendix C + Sec. 5

---

## Reviewer 4 (R4) - 评分 5/5
- ✅ R4-1: Jacobian计算和协方差更新 → Sec. 3.3, Eq.(8) vs (9) 分离
- ✅ R4-2: 掩码影响；全景指标 → Sec. 4.1 + Table 2 (无掩码AUSE) + Fig.7-8 Table 3
- ✅ **R4-3: 多对象和单对象限制** → **新增**："single-object scenarios...selects one next-best view per iteration"
- ✅ **R4-3b: 图3符号不一致** → **新增**："reviewed and corrected notation alignment in Figure 3"

---

## Reviewer 5 (R5) - 评分 6/6
- ✅ R5-1: 缺失定义 (ℓ_t, λ) → Sec. 3.3, Eq.(8)-(9) 主文本定义
- ✅ R5-2: 参考文献清洁度和指标 → Sec. 4.1 "Metrics" + References更新
- ✅ **R5-3: 小文本编辑错误** → **新增**：page 2 PUP 3D-GS citation, page 6编辑错误和"feed identical...For fairness", page 9结论截断修复

---

## 最终状态

**总计点数：** 25个
- **✅ 已覆盖：** 25个 (100%)
- **❌ 未覆盖：** 0个

### 新增的补充（本次）：
1. R1-5: POp-GS/UNG-GS在Sec. 2.2和Sec. 4.1已讨论，cover letter明确指向
2. R2-1b: 应用示例（cultural heritage preservation, industrial inspection, AR）在Introduction添加
3. R4-3: 单对象限制明确说明（一次选择一个对象）
4. R4-3b: 图3符号对齐修复
5. R5-3: 小编辑错误列表和修复

---

## Cover Letter 已准备好上传！

**最后确认清单：**
- ✅ 所有Summary Review要求已覆盖
- ✅ 所有5个Reviewer（R1-R5）的所有点都已回应
- ✅ 每个回应都包含具体的论文位置（Sec, Eq, Fig, Table）
- ✅ 新增补充完整且有逻辑
- ✅ 格式清晰，易于审阅者查找
