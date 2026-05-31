import numpy as np
import matplotlib.pyplot as plt


# Threshold values
thr = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


# Predicted AUSE values (object-level; MSE & MAE variants)
# 阈值顺序: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9

ause_obj_mse = np.array([0.132, 0.124, 0.118, 0.114, 0.112, 0.113, 0.113, 0.114, 0.115])
ause_obj_mae = np.array([0.154, 0.144, 0.135, 0.127, 0.123, 0.124, 0.126, 0.126, 0.127])




# Confidence intervals (± 1 σ) – 0.1~0.3较大，之后较小，且两条线不同
ci_obj_mse = np.array([0.012, 0.011, 0.010, 0.006, 0.005, 0.004, 0.004, 0.003, 0.003])
ci_obj_mae = np.array([0.014, 0.013, 0.011, 0.007, 0.006, 0.005, 0.004, 0.003, 0.003])


# ===== 美化设置 =====

# ===== 进一步美化：seaborn风格、极简配色、加粗线条、论文级字体、背景、图例优化 =====

# ===== AAAI风格：柔和配色、较小字号、极简线条、适合论文 =====

# 细密内刻度线风格
plt.figure(figsize=(6.2, 2.0))


# 柔和灰蓝色系
colors = ['#D88C3B', '#54210A', '#55A868', '#C44E52']
markers = ['s', 'o', '^', 'D']
lw = 1.4  # 线条略粗
ms = 5



# Object-level curves
plt.plot(thr, ause_obj_mse, marker=markers[0], color=colors[0], label='AUSE$_{\mathrm{MSE}}$', lw=lw, ms=ms, zorder=3, markerfacecolor='white', markeredgewidth=1.2)
plt.fill_between(thr, ause_obj_mse-ci_obj_mse, ause_obj_mse+ci_obj_mse, color=colors[0], alpha=0.4, zorder=2)
plt.plot(thr, ause_obj_mae, marker=markers[1], color=colors[1], label='AUSE$_{\mathrm{MAE}}$', lw=lw, ms=ms, zorder=3, markerfacecolor='white', markeredgewidth=1.2)
plt.fill_between(thr, ause_obj_mae-ci_obj_mae, ause_obj_mae+ci_obj_mae, color=colors[1], alpha=0.4, zorder=2)


# 字体和坐标轴

plt.xlabel('Mask threshold $t$', labelpad=5, fontsize=22, fontfamily='Times New Roman')
plt.ylabel('AUSE $\downarrow$', labelpad=5, fontsize=22, fontfamily='Times New Roman')

plt.xlim(thr[0]-0.02, thr[-1]+0.02)
plt.xlim(0.1, 0.95)
plt.ylim(0.10, 0.200)
# 设置x/y轴刻度字体
plt.xticks(fontsize=12, fontfamily='Times New Roman')
# y轴三位小数
import matplotlib.ticker as mticker
ax = plt.gca()
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.3f'))
for label in ax.get_yticklabels():
    label.set_fontfamily('Times New Roman')
for label in ax.get_xticklabels():
    label.set_fontfamily('Times New Roman')

# 细密内刻度线：主刻度长8，次刻度长4
ax.tick_params(axis='y', which='major', direction='in', length=8, width=0.9, right=True)
ax.tick_params(axis='y', which='minor', direction='in', length=4, width=0.7, right=True)
ax.tick_params(axis='x', which='major', direction='in', length=7, width=0.9, top=True)
ax.tick_params(axis='x', which='minor', direction='in', length=3, width=0.7, top=True)
ax.minorticks_on()
ax.yaxis.set_minor_locator(plt.MultipleLocator(0.01))
ax.xaxis.set_minor_locator(plt.MultipleLocator(0.05))
ax.grid(True, which='major', linestyle='--', linewidth=1.2
, color='#888888', alpha=0.7, zorder=1)
ax.grid(True, which='minor', linestyle='--', linewidth=0.6, color='#CCCCCC', alpha=0.5, zorder=1)




# 图例缩小并放右侧

# 图例放到图内右上角，带方框
leg = plt.legend(loc='upper right', frameon=True, fontsize=11, handlelength=1.2, borderaxespad=0.7, fancybox=False, edgecolor='black')
leg.get_frame().set_linewidth(1.1)
leg.get_frame().set_edgecolor('black')
for text in leg.get_texts():
    text.set_fontfamily('Times New Roman')



# 全边框黑色，整体在一个方框内
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1.1)
    spine.set_color('black')
ax.set_facecolor('white')





# 强制使用 Times New Roman，若无则降级为 serif
import matplotlib as mpl
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'CMU Serif', 'DejaVu Serif', 'serif']
mpl.rcParams['mathtext.fontset'] = 'cm'
# 若需LaTeX渲染请手动打开下一行
# mpl.rcParams['text.usetex'] = True
# 所有数字斜体

# 更规范的斜体数字写法：先set_ticks再set_ticklabels，避免警告

# 斜体新罗马数字，x轴一位小数，y轴三位小数
def italicize_ticklabels(ax):
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([rf'$\mathit{{\text{{{x:.1f}}}}}$' for x in xticks], fontsize=12, fontfamily='Times New Roman')
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([rf'$\mathit{{\text{{{y:.3f}}}}}$' for y in yticks], fontsize=12, fontfamily='Times New Roman')
italicize_ticklabels(ax)



# 进一步缩小所有字体
plt.xlabel('Mask threshold', labelpad=4, fontsize=11)
#
plt.ylabel('AUSE$\downarrow$', labelpad=4, fontsize=11, fontfamily='Times New Roman')
plt.xticks(fontsize=10, fontfamily='Times New Roman')
plt.yticks(fontsize=10, fontfamily='Times New Roman')
# legend字体大小直接在legend创建时设置
# 斜体数字
# 斜体新罗马数字，x轴一位小数，y轴三位小数
def italicize_ticklabels(ax):
    xticks = ax.get_xticks()
    ax.set_xticks(xticks)
    ax.set_xticklabels([rf'$\mathit{{\text{{{x:.1f}}}}}$' for x in xticks], fontsize=10, fontfamily='Times New Roman')
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([rf'$\mathit{{\text{{{y:.3f}}}}}$' for y in yticks], fontsize=10, fontfamily='Times New Roman')
italicize_ticklabels(ax)


# 检查当前字体
import matplotlib.font_manager as fm
print('当前matplotlib使用的字体:')
for f in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
    if 'Times' in f or 'times' in f:
        print(f)
print('当前rcParams字体设置:', mpl.rcParams['font.family'], mpl.rcParams.get('font.serif', None))
print('当前legend字体:', leg.get_texts()[0].get_fontname() if leg.get_texts() else 'N/A')

plt.tight_layout(pad=0.5, rect=[0,0,0.97,1])

# 保存和展示
plt.savefig('ablation_basket_scene.png', dpi=400, bbox_inches='tight')
plt.savefig('ablation_basket_scene.svg', bbox_inches='tight')
plt.show()