"""
GPS-TTO (Test-Time Optimization) 数据流图
=========================================
基于 LongStream 代码仓中的实现逻辑，使用 graphviz 绘制学术级别的算法数据流图。

核心逻辑总结:
1. 前向推理流 (LongStream Standard Flow):
   - RGB 图像序列 → DINOv2 ViT 视觉编码器（patch_embed）→ STreamAggregator 聚合
   - Scale Token 在 Aggregator 阶段注入，与 patch token 拼接后一并处理
   - Aggregator 输出特征 → CameraHead / RelPoseHead 预测位姿编码 (pose_enc)
   - Scale Head 从 scale_token 输出特征回归 scale_factor (exp(logit))
   - scale_factor 乘以位姿平移分量、点云、深度 → 全局尺度对齐

2. GPS-TTO 反向优化流:
   - GPS 帧间位移约束: 对比预测相机中心位移与 GPS 位移
   - 可微分相机中心恢复: pose_enc → quat_to_mat → 绝对位姿累积 → C = -R^T @ t
   - 多尺度位移 Huber Loss (含方向约束、端点约束)
   - AdamW 优化器仅更新 scale_token (1-D nn.Parameter)
   - 梯度阻断: 模型所有参数 requires_grad=False，仅 scale_token 为 True
     - ViT Backbone (patch_embed): Frozen ❄
     - STreamAggregator Trunk: Frozen ❄
     - CameraHead / RelPoseHead: Frozen ❄
     - Scale Token: Trainable ✓ → 梯度仅回传至此

依赖: pip install graphviz
运行: python gps_tto_flowchart.py
输出: gps_tto_flowchart.pdf / gps_tto_flowchart.png
"""

import graphviz

# ═══════════════════════════════════════════════════════════════════════════
#  配色体系 —— 仿参考图的马卡龙色/柔和粉彩学术风格
# ═══════════════════════════════════════════════════════════════════════════

# 子图（Subgraph）背景色 —— 极浅马卡龙色 + 虚线边框
BG_INPUT       = "#EDE7F6"   # 极浅紫（输入区域）
BG_ENCODER     = "#E3F2FD"   # 极浅蓝（编码器区域）
BG_GEOMETRY    = "#FFF8E1"   # 极浅黄（几何预测区域）
BG_OUTPUT      = "#F3E5F5"   # 极浅薰衣草（输出区域）
BG_GPS         = "#E8F5E9"   # 极浅绿（GPS 先验区域）
BG_TTO         = "#FBE9E7"   # 极浅珊瑚（TTO 优化区域）

# 子图边框色
BORDER_INPUT   = "#7E57C2"   # 中等紫
BORDER_ENCODER = "#42A5F5"   # 中等蓝
BORDER_GEOMETRY = "#FFB300"  # 中等琥珀
BORDER_OUTPUT  = "#AB47BC"   # 中等紫罗兰
BORDER_GPS     = "#66BB6A"   # 中等绿
BORDER_TTO     = "#EF5350"   # 中等红

# 节点填充色 —— 各区域对应的饱和色
NODE_INPUT     = "#5E35B1"   # 深紫
NODE_ENCODER   = "#1E88E5"   # 深蓝
NODE_GEOMETRY  = "#F9A825"   # 深琥珀/土黄
NODE_OUTPUT    = "#8E24AA"   # 深紫罗兰
NODE_GPS       = "#43A047"   # 深绿
NODE_TTO       = "#E53935"   # 深红

# 特殊节点
NODE_SCALE_TOKEN = "#FF6F00" # 鲜橙 —— Scale Token（唯一可训练参数）
NODE_FROZEN      = "#78909C" # 蓝灰 —— 表示冻结参数

# 连接线颜色
EDGE_FORWARD   = "#37474F"   # 深灰 —— 前向数据流
EDGE_GRADIENT  = "#D32F2F"   # 亮红 —— 梯度反传路径
EDGE_INJECT    = "#FF6F00"   # 鲜橙 —— Scale Token 注入路径
EDGE_GPS_IN    = "#2E7D32"   # 深绿 —— GPS 数据输入

# 侧边栏标注颜色
SIDEBAR_COLOR  = "#C62828"   # 红色字体

# 节点字体颜色
FONT_WHITE     = "#FFFFFF"
FONT_DARK      = "#212121"

# ═══════════════════════════════════════════════════════════════════════════
#  构建 Graphviz 有向图
# ═══════════════════════════════════════════════════════════════════════════

dot = graphviz.Digraph(
    name="GPS_TTO_Flowchart",
    format="png",
    engine="dot",
)

# 全局属性
dot.attr(
    rankdir="TB",            # 从上到下排列
    bgcolor="#FFFFFF",       # 白色背景
    fontname="Microsoft YaHei",
    fontsize="14",
    dpi="300",               # 高分辨率
    pad="0.5",
    nodesep="0.6",
    ranksep="0.7",
    splines="ortho",         # 正交连接线
    compound="true",         # 允许子图间连线
)

dot.attr("node",
    shape="box",
    style="filled,rounded",
    fontname="Microsoft YaHei",
    fontsize="11",
    fontcolor=FONT_WHITE,
    penwidth="0",            # 无边框
    margin="0.15,0.08",
    height="0.5",
)

dot.attr("edge",
    fontname="Microsoft YaHei",
    fontsize="9",
    arrowsize="0.8",
)


# ═══════════════════════════════════════════════════════════════════════════
#  侧边栏标注节点（放在最左侧，不在任何子图内）
# ═══════════════════════════════════════════════════════════════════════════

sidebar_attrs = dict(
    shape="plaintext",
    fontname="Microsoft YaHei Bold",
    fontsize="13",
    fontcolor=SIDEBAR_COLOR,
    style="bold",
    width="0",
    height="0",
)

dot.node("sidebar_input",    "数据\n输入",     **sidebar_attrs)
dot.node("sidebar_encoder",  "视觉\n编码",   **sidebar_attrs)
dot.node("sidebar_geometry", "几何\n预测",   **sidebar_attrs)
dot.node("sidebar_output",   "融合\n输出",   **sidebar_attrs)
dot.node("sidebar_gps",      "GPS\n先验",    **sidebar_attrs)
dot.node("sidebar_tto",      "尺度\nTTO",    **sidebar_attrs)

# ═══════════════════════════════════════════════════════════════════════════
#  子图 1：数据输入区
# ═══════════════════════════════════════════════════════════════════════════

with dot.subgraph(name="cluster_input") as c:
    c.attr(
        label="数据输入层",
        style="dashed,rounded",
        color=BORDER_INPUT,
        bgcolor=BG_INPUT,
        fontname="Microsoft YaHei Bold",
        fontsize="12",
        fontcolor=BORDER_INPUT,
        penwidth="2.0",
        margin="16",
    )
    c.node("rgb_seq",     "RGB 图像序列\n[B, S, C, H, W]",   fillcolor=NODE_INPUT)
    c.node("keyframe",    "关键帧选择\n(KeyframeSelector)",   fillcolor=NODE_INPUT)
    c.node("kv_cache",    "KV 缓存管理\n(Streaming Cache)",   fillcolor=NODE_INPUT)

# ═══════════════════════════════════════════════════════════════════════════
#  子图 2：视觉编码区（ViT Backbone + Aggregator）
# ═══════════════════════════════════════════════════════════════════════════

with dot.subgraph(name="cluster_encoder") as c:
    c.attr(
        label="视觉编码层",
        style="dashed,rounded",
        color=BORDER_ENCODER,
        bgcolor=BG_ENCODER,
        fontname="Microsoft YaHei Bold",
        fontsize="12",
        fontcolor=BORDER_ENCODER,
        penwidth="2.0",
        margin="16",
    )
    c.node("vit_backbone", "DINOv2 ViT-L 视觉编码器\n(patch_embed)  ❄ Frozen",
           fillcolor=NODE_FROZEN)
    c.node("scale_token",  "尺度表征向量\n(Scale Token)  ✎ 可训练",
           fillcolor=NODE_SCALE_TOKEN, fontcolor=FONT_WHITE, penwidth="3",
           style="filled,rounded,bold", color="#E65100")
    c.node("aggregator",   "STreamAggregator 聚合器\n(Transformer Blocks)  ❄ Frozen",
           fillcolor=NODE_FROZEN)

# ═══════════════════════════════════════════════════════════════════════════
#  子图 3：几何预测区
# ═══════════════════════════════════════════════════════════════════════════

with dot.subgraph(name="cluster_geometry") as c:
    c.attr(
        label="几何预测层",
        style="dashed,rounded",
        color=BORDER_GEOMETRY,
        bgcolor=BG_GEOMETRY,
        fontname="Microsoft YaHei Bold",
        fontsize="12",
        fontcolor=BORDER_GEOMETRY,
        penwidth="2.0",
        margin="16",
    )
    c.node("rel_pose_head", "相对位姿估计头\n(RelPoseHead)  ❄ Frozen",
           fillcolor=NODE_FROZEN)
    c.node("scale_head",    "尺度回归头\n(Scale Head → exp(logit))",
           fillcolor=NODE_GEOMETRY)
    c.node("depth_head",    "深度预测头\n(DPTHead)  ❄ Frozen",
           fillcolor=NODE_FROZEN)
    c.node("point_head",    "点云预测头\n(DPTHead)  ❄ Frozen",
           fillcolor=NODE_FROZEN)

# ═══════════════════════════════════════════════════════════════════════════
#  子图 4：融合输出区
# ═══════════════════════════════════════════════════════════════════════════

with dot.subgraph(name="cluster_output") as c:
    c.attr(
        label="尺度融合输出",
        style="dashed,rounded",
        color=BORDER_OUTPUT,
        bgcolor=BG_OUTPUT,
        fontname="Microsoft YaHei Bold",
        fontsize="12",
        fontcolor=BORDER_OUTPUT,
        penwidth="2.0",
        margin="16",
    )
    c.node("scale_multiply", "全局尺度乘法融合\npose.t × s, pts3d × s, depth × s",
           fillcolor=NODE_OUTPUT)
    c.node("output_pose",    "校准位姿输出\n(Scaled Pose)",
           fillcolor=NODE_OUTPUT)
    c.node("output_3d",      "校准三维重建\n(3DGS / NeRF 点云)",
           fillcolor=NODE_OUTPUT)

# ═══════════════════════════════════════════════════════════════════════════
#  子图 5：GPS 先验区
# ═══════════════════════════════════════════════════════════════════════════

with dot.subgraph(name="cluster_gps") as c:
    c.attr(
        label="GPS 位移先验",
        style="dashed,rounded",
        color=BORDER_GPS,
        bgcolor=BG_GPS,
        fontname="Microsoft YaHei Bold",
        fontsize="12",
        fontcolor=BORDER_GPS,
        penwidth="2.0",
        margin="16",
    )
    c.node("gps_input",      "GPS 坐标序列\n[S, 3] 世界坐标",
           fillcolor=NODE_GPS)
    c.node("gps_disp",       "GPS 帧间位移约束\n多尺度步长 (stride)",
           fillcolor=NODE_GPS)

# ═══════════════════════════════════════════════════════════════════════════
#  子图 6：TTO 尺度优化区
# ═══════════════════════════════════════════════════════════════════════════

with dot.subgraph(name="cluster_tto") as c:
    c.attr(
        label="流形约束尺度 TTO\n(Manifold-Constrained Scale TTO)",
        style="dashed,rounded",
        color=BORDER_TTO,
        bgcolor=BG_TTO,
        fontname="Microsoft YaHei Bold",
        fontsize="12",
        fontcolor=BORDER_TTO,
        penwidth="2.0",
        margin="16",
    )
    c.node("camera_center",  "可微分相机中心恢复\nC = -R^T·t (quat_to_mat)",
           fillcolor=NODE_TTO)
    c.node("pred_disp",      "预测帧间位移\n||C_i - C_j||",
           fillcolor=NODE_TTO)
    c.node("scale_loss",     "多尺度位移损失\nHuber Loss + 方向/端点约束",
           fillcolor=NODE_TTO)
    c.node("optimizer",      "AdamW 优化器\n梯度裁剪 + 早停策略",
           fillcolor=NODE_TTO)


# ═══════════════════════════════════════════════════════════════════════════
#  前向数据流 —— 深色实线粗箭头
# ═══════════════════════════════════════════════════════════════════════════

forward_attrs = dict(
    color=EDGE_FORWARD,
    penwidth="2.0",
    arrowsize="0.9",
    style="solid",
)

# 输入 → 编码器
dot.edge("rgb_seq",       "vit_backbone", label="  图像输入", **forward_attrs)
dot.edge("rgb_seq",       "keyframe",     **forward_attrs)
dot.edge("keyframe",      "aggregator",   label="  is_keyframe\n  keyframe_idx", **forward_attrs)
dot.edge("kv_cache",      "aggregator",   label="  缓存复用", **forward_attrs)

# 编码器内部
dot.edge("vit_backbone",  "aggregator",   label="  patch tokens", **forward_attrs)
dot.edge("scale_token",   "aggregator",
         label="  拼接注入",
         color=EDGE_INJECT, penwidth="2.5", style="solid",
         arrowsize="0.9", fontcolor=EDGE_INJECT)

# 编码器 → 几何预测
dot.edge("aggregator",    "rel_pose_head", label="  聚合特征", **forward_attrs)
dot.edge("aggregator",    "scale_head",    label="  scale_token\n  输出特征",
         color=EDGE_INJECT, penwidth="2.5", style="solid",
         arrowsize="0.9", fontcolor=EDGE_INJECT)
dot.edge("aggregator",    "depth_head",    **forward_attrs)
dot.edge("aggregator",    "point_head",    **forward_attrs)

# 几何预测 → 融合输出
dot.edge("rel_pose_head", "scale_multiply", label="  pose_enc", **forward_attrs)
dot.edge("scale_head",    "scale_multiply", label="  scale_factor\n  = exp(logit)",
         color=EDGE_INJECT, penwidth="2.5", style="solid",
         arrowsize="0.9", fontcolor=EDGE_INJECT)
dot.edge("depth_head",    "scale_multiply", label="  depth", **forward_attrs)
dot.edge("point_head",    "scale_multiply", label="  pts3d", **forward_attrs)

# 融合输出
dot.edge("scale_multiply", "output_pose", **forward_attrs)
dot.edge("scale_multiply", "output_3d",   **forward_attrs)


# ═══════════════════════════════════════════════════════════════════════════
#  GPS 输入流
# ═══════════════════════════════════════════════════════════════════════════

dot.edge("gps_input", "gps_disp",
         label="  位移计算",
         color=EDGE_GPS_IN, penwidth="2.0", style="solid",
         arrowsize="0.9", fontcolor=EDGE_GPS_IN)


# ═══════════════════════════════════════════════════════════════════════════
#  TTO 前向计算路径 (位姿 → 相机中心 → 位移 → Loss)
# ═══════════════════════════════════════════════════════════════════════════

tto_forward_attrs = dict(
    color=EDGE_GRADIENT,
    penwidth="2.0",
    style="solid",
    fontcolor=EDGE_GRADIENT,
    arrowsize="0.9",
)

dot.edge("rel_pose_head", "camera_center", label="  pose_enc\n  (保留计算图)", **tto_forward_attrs)
dot.edge("camera_center", "pred_disp",     label="  预测位移", **tto_forward_attrs)
dot.edge("pred_disp",     "scale_loss",    label="  pred_disp", **tto_forward_attrs)
dot.edge("gps_disp",      "scale_loss",    label="  GPS 真值位移",
         color=EDGE_GPS_IN, penwidth="2.0", style="solid",
         arrowsize="0.9", fontcolor=EDGE_GPS_IN)


# ═══════════════════════════════════════════════════════════════════════════
#  梯度反传路径 —— 红色粗虚线，附带标注
# ═══════════════════════════════════════════════════════════════════════════

grad_attrs = dict(
    color=EDGE_GRADIENT,
    penwidth="2.5",
    style="dashed",
    fontcolor=EDGE_GRADIENT,
    arrowsize="1.0",
)

# Loss → Optimizer
dot.edge("scale_loss", "optimizer",
         label="  ∂L/∂θ 反传",
         **grad_attrs)

# Optimizer → Scale Token（梯度最终目标）
dot.edge("optimizer", "scale_token",
         label="  ⚡ 仅更新 Scale Token\n  (Manifold-Constrained)",
         color="#D32F2F", penwidth="3.0", style="dashed,bold",
         fontcolor="#D32F2F", arrowsize="1.2")

# 标注：梯度穿越路径
dot.edge("scale_loss", "pred_disp",
         label="  ∇ 梯度回传",
         dir="back", **grad_attrs)

dot.edge("pred_disp", "camera_center",
         label="  ∇ 可微分路径",
         dir="back", **grad_attrs)

dot.edge("camera_center", "rel_pose_head",
         label="  ∇ 穿越 (Frozen)",
         dir="back",
         color=EDGE_GRADIENT, penwidth="2.0", style="dashed",
         fontcolor="#78909C",  # 灰色标注表示穿越冻结层
         arrowsize="0.8")

dot.edge("rel_pose_head", "aggregator",
         label="  ∇ 穿越 (Frozen)",
         dir="back",
         color=EDGE_GRADIENT, penwidth="2.0", style="dashed",
         fontcolor="#78909C",
         arrowsize="0.8")

dot.edge("aggregator", "scale_token",
         label="  ∇ 梯度终点 → Scale Token",
         dir="back",
         color="#D32F2F", penwidth="3.0", style="dashed,bold",
         fontcolor="#D32F2F",
         arrowsize="1.0")


# ═══════════════════════════════════════════════════════════════════════════
#  侧边栏对齐 —— 使用隐形边固定侧边栏位置
# ═══════════════════════════════════════════════════════════════════════════

invis_attrs = dict(style="invis", weight="10")

dot.edge("sidebar_input",    "sidebar_encoder",  **invis_attrs)
dot.edge("sidebar_encoder",  "sidebar_geometry",  **invis_attrs)
dot.edge("sidebar_geometry", "sidebar_output",    **invis_attrs)
dot.edge("sidebar_output",   "sidebar_gps",       **invis_attrs)
dot.edge("sidebar_gps",      "sidebar_tto",       **invis_attrs)

# 侧边栏与主图对齐（同一 rank）
with dot.subgraph(name="rank_input") as s:
    s.attr(rank="same")
    s.node("sidebar_input")
    s.node("rgb_seq")
    s.node("keyframe")
    s.node("kv_cache")

with dot.subgraph(name="rank_encoder") as s:
    s.attr(rank="same")
    s.node("sidebar_encoder")
    s.node("vit_backbone")
    s.node("scale_token")
    s.node("aggregator")

with dot.subgraph(name="rank_geometry") as s:
    s.attr(rank="same")
    s.node("sidebar_geometry")
    s.node("rel_pose_head")
    s.node("scale_head")
    s.node("depth_head")
    s.node("point_head")

with dot.subgraph(name="rank_output") as s:
    s.attr(rank="same")
    s.node("sidebar_output")
    s.node("scale_multiply")
    s.node("output_pose")
    s.node("output_3d")

with dot.subgraph(name="rank_gps") as s:
    s.attr(rank="same")
    s.node("sidebar_gps")
    s.node("gps_input")
    s.node("gps_disp")
    s.node("camera_center")
    s.node("pred_disp")

with dot.subgraph(name="rank_tto") as s:
    s.attr(rank="same")
    s.node("sidebar_tto")
    s.node("scale_loss")
    s.node("optimizer")


# ═══════════════════════════════════════════════════════════════════════════
#  图例节点
# ═══════════════════════════════════════════════════════════════════════════

with dot.subgraph(name="cluster_legend") as c:
    c.attr(
        label="图例 (Legend)",
        style="dashed,rounded",
        color="#BDBDBD",
        bgcolor="#FAFAFA",
        fontname="Microsoft YaHei Bold",
        fontsize="11",
        fontcolor="#757575",
        penwidth="1.5",
        margin="12",
    )
    c.node("leg_frozen",  "❄ Frozen 参数\n(requires_grad=False)",
           fillcolor=NODE_FROZEN, fontsize="9")
    c.node("leg_trainable", "✎ 可训练参数\n(requires_grad=True)",
           fillcolor=NODE_SCALE_TOKEN, fontsize="9")
    c.node("leg_forward",  "━━ 前向数据流\n(实线粗箭头)",
           fillcolor=EDGE_FORWARD, fontsize="9")
    c.node("leg_grad",    "╌╌ 梯度反传流\n(红色虚线箭头)",
           fillcolor=EDGE_GRADIENT, fontsize="9")

    # 图例内部排列
    c.edge("leg_frozen", "leg_trainable", style="invis")
    c.edge("leg_trainable", "leg_forward", style="invis")
    c.edge("leg_forward", "leg_grad", style="invis")


# ═══════════════════════════════════════════════════════════════════════════
#  渲染输出
# ═══════════════════════════════════════════════════════════════════════════

import os
output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, "gps_tto_flowchart")

# 1) 始终保存 .gv 源文件（不依赖 Graphviz 二进制）
gv_source_path = output_path + ".gv"
dot.save(gv_source_path)
print(f"[OK] Graphviz 源文件已保存: {gv_source_path}")

# 2) 尝试渲染 PNG / PDF / SVG
try:
    dot.format = "png"
    dot.render(output_path, cleanup=True, view=False)
    print(f"[OK] PNG 已生成: {output_path}.png")

    dot.format = "pdf"
    dot.render(output_path + "_pdf", cleanup=True, view=False)
    print(f"[OK] PDF 已生成: {output_path}_pdf.pdf")

    dot.format = "svg"
    dot.render(output_path + "_svg", cleanup=True, view=False)
    print(f"[OK] SVG 已生成: {output_path}_svg.svg")

except Exception as exc:
    print(f"\n[!] 渲染失败: {exc}")
    print("[!] Graphviz 二进制未安装或不在 PATH 中。")
    print("[!] 请先安装 Graphviz:")
    print("    Windows: winget install Graphviz.Graphviz")
    print("             或从 https://graphviz.org/download/ 下载安装")
    print("    安装后重启终端，然后重新运行本脚本。")
    print(f"[!] .gv 源文件已保存至: {gv_source_path}")
    print("[!] 你也可以将 .gv 文件粘贴到在线渲染器: https://dreampuf.github.io/GraphvizOnline/")
