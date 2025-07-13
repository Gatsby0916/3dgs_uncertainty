import argparse
from arguments import ModelParams, PipelineParams
from scene import Scene
from scene.gaussian_model import GaussianModel

# 0. 手动构造命令行参数列表
arg_list = [
    "-s", "data/LF/basket",   # 数据集根目录
    "--depths", "none"        # <<—— 关键：禁用所有 depth 读写
]

# 1. 解析参数
parser = argparse.ArgumentParser()
model = ModelParams(parser, sentinel=True)
pipe  = PipelineParams(parser)
args  = parser.parse_args(arg_list)

# 2. 加载 Scene 并指定 checkpoint iter
scene = Scene(args, GaussianModel(3), load_iteration=3000)

# 3. 打印 Fisher-Inv 对角向量形状
finv = scene.gaussians.get_fisher_inv_diag()
print("✓  Finv diag shape =", finv.shape)
