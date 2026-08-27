from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(BOLD if bold else FONT, size)


def rounded_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    *,
    fill: str,
    outline: str,
) -> None:
    draw.rounded_rectangle(xy, radius=12, fill=fill, outline=outline, width=3)
    x1, y1, x2, _ = xy
    draw.text((x1 + 22, y1 + 18), title, font=font(28, bold=True), fill="#17231f")
    draw.multiline_text(
        (x1 + 22, y1 + 62),
        subtitle,
        font=font(18),
        fill="#4f615b",
        spacing=6,
    )
    draw.line((x1 + 22, y1 + 54, x2 - 22, y1 + 54), fill=outline, width=2)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str = "#52645e",
    width: int = 4,
) -> None:
    draw.line((*start, *end), fill=color, width=width)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex > sx else -1
        points = [(ex, ey), (ex - 14 * direction, ey - 8), (ex - 14 * direction, ey + 8)]
    else:
        direction = 1 if ey > sy else -1
        points = [(ex, ey), (ex - 8, ey - 14 * direction), (ex + 8, ey - 14 * direction)]
    draw.polygon(points, fill=color)


def architecture() -> None:
    image = Image.new("RGB", (1600, 940), "#f7faf9")
    draw = ImageDraw.Draw(image)
    draw.text((70, 46), "HarnessVLN · Agent 主导架构", font=font(40, bold=True), fill="#14211d")
    draw.text(
        (72, 105),
        "Runner 调度完整任务；Agent 通过类型化工具自主观察、调用 VLN、移动并停止。",
        font=font(21),
        fill="#50635d",
    )

    rounded_box(draw, (70, 190, 390, 350), "Bench", "加载 Task\n持有私有真值与评分器", fill="#eef4fb", outline="#4a7dab")
    rounded_box(draw, (470, 190, 790, 350), "Runner", "有界调度完整 Task\n不参与 observe-act", fill="#f5f0e6", outline="#a47627")
    rounded_box(draw, (870, 190, 1530, 350), "NavigationHarness", "单 Task 生命周期、终止竞争、清理顺序", fill="#eef7f3", outline="#32806a")

    rounded_box(draw, (70, 470, 430, 675), "Agent Core", "唯一决策主体\n自由 loop 或固定 workflow\n主动调用 nav.stop", fill="#e8f5f0", outline="#176b58")
    rounded_box(draw, (520, 470, 920, 675), "ToolBus", "JSON Schema · 权限白名单\n写屏障 · 审计事件", fill="#f4f3fb", outline="#665c9e")
    rounded_box(draw, (1010, 430, 1530, 565), "VLN plugin", "完整模型内部循环与状态；反向调用导航工具", fill="#f8eeee", outline="#a55258")
    rounded_box(draw, (1010, 595, 1250, 790), "Environment", "标准观测与动作\n映射模拟器/真机", fill="#edf5fb", outline="#39759f")
    rounded_box(draw, (1290, 595, 1530, 790), "Memory", "landmark 查询与写入\n可跨 Task 持久化", fill="#f7f1e8", outline="#9a661c")

    arrow(draw, (390, 270), (470, 270))
    arrow(draw, (790, 270), (870, 270))
    arrow(draw, (1200, 350), (1200, 420))
    arrow(draw, (430, 575), (510, 575), color="#176b58")
    arrow(draw, (920, 520), (1000, 500), color="#665c9e")
    arrow(draw, (920, 600), (1000, 680), color="#665c9e")
    arrow(draw, (920, 625), (1280, 690), color="#665c9e")
    arrow(draw, (1010, 540), (930, 540), color="#a55258")

    draw.text((82, 835), "控制权", font=font(18, bold=True), fill="#176b58")
    draw.text((160, 835), "Agent 决定何时观察、何时行动、何时结束。", font=font(18), fill="#50635d")
    draw.text((82, 875), "隔离边界", font=font(18, bold=True), fill="#665c9e")
    draw.text((180, 875), "ToolBus 让组件只看到自己声明过的工具。", font=font(18), fill="#50635d")
    image.save(PUBLIC / "architecture-overview.png", optimize=True)


def lifecycle() -> None:
    image = Image.new("RGB", (1600, 720), "#f7faf9")
    draw = ImageDraw.Draw(image)
    draw.text((70, 42), "单 Task 生命周期", font=font(40, bold=True), fill="#14211d")
    draw.text((72, 100), "启动顺序与停止顺序并不对称：先封住运动，再关闭运动命令生产者。", font=font(21), fill="#50635d")

    stages = [
        ("01", "Environment.start", "注册 observe / move / goal.finish", "#39759f"),
        ("02", "Memory.start", "注册 spatial.search / remember", "#9a661c"),
        ("03", "VLN.start", "校验 requirements，注册 job tools", "#a55258"),
        ("04", "Agent.run", "唯一一次；内部自主循环", "#176b58"),
        ("05", "Terminal", "stop / env terminal / timeout / exception", "#665c9e"),
    ]
    x = 70
    for number, title, subtitle, color in stages:
        draw.rounded_rectangle((x, 190, x + 270, 360), radius=10, fill="#ffffff", outline=color, width=3)
        draw.text((x + 18, 208), number, font=font(18, bold=True), fill=color)
        draw.text((x + 18, 245), title, font=font(23, bold=True), fill="#17231f")
        draw.multiline_text((x + 18, 292), subtitle, font=font(16), fill="#50635d", spacing=4)
        if x + 270 < 1500:
            arrow(draw, (x + 270, 275), (x + 295, 275), color="#75847f", width=3)
        x += 300

    draw.text((70, 425), "停止与回收", font=font(25, bold=True), fill="#a33e45")
    cleanup = [
        "关闭 ToolBus 写入口",
        "停止 Environment，阻断原生运动",
        "停止 VLN，取消后台 Job",
        "等待写调用排空",
        "持久化 Memory",
        "读取 Environment.result",
    ]
    x = 70
    for index, label in enumerate(cleanup, 1):
        width = 230 if index < 6 else 260
        draw.rounded_rectangle((x, 480, x + width, 585), radius=8, fill="#fff7f5", outline="#c36d63", width=2)
        draw.text((x + 14, 496), f"{index:02d}", font=font(16, bold=True), fill="#a33e45")
        draw.multiline_text((x + 14, 528), label, font=font(16), fill="#4f5f5a", spacing=3)
        if index < len(cleanup):
            arrow(draw, (x + width, 532), (x + width + 20, 532), color="#a66a62", width=3)
        x += width + 28
    draw.text((72, 632), "关键不变量：nav.stop 之后不能再接受新的写工具；清理错误独立记录，不覆盖主失败。", font=font(20), fill="#50635d")
    image.save(PUBLIC / "task-lifecycle.png", optimize=True)


def mark() -> None:
    image = Image.new("RGBA", (192, 192), "#176b58")
    draw = ImageDraw.Draw(image)
    draw.line((38, 136, 70, 86, 112, 112, 153, 48), fill="#d8f2e9", width=12, joint="curve")
    for x, y in ((38, 136), (70, 86), (112, 112), (153, 48)):
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill="#ffffff")
    draw.text((40, 30), "H", font=font(54, bold=True), fill="#ffffff")
    image.save(PUBLIC / "harnessvln-mark.png", optimize=True)


if __name__ == "__main__":
    PUBLIC.mkdir(parents=True, exist_ok=True)
    architecture()
    lifecycle()
    mark()
