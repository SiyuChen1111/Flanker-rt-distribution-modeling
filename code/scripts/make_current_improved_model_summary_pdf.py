#!/usr/bin/env python3
"""Create a concise Chinese PDF summary of the improved R5 model."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageTemplate,
    Paragraph,
    PageBreak,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT_ROOT / "output/pdf/current_improved_r5_model_summary_tables.pdf"
FONT_PATH = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
FONT_NAME = "ArialUnicode"

NAVY = colors.HexColor("#19324D")
BLUE = colors.HexColor("#4472C4")
LIGHT_BLUE = colors.HexColor("#EAF1FA")
ORANGE = colors.HexColor("#E69F00")
LIGHT_ORANGE = colors.HexColor("#FFF3D6")
LIGHT_GRAY = colors.HexColor("#F3F5F7")
MID_GRAY = colors.HexColor("#66727F")
GRID = colors.HexColor("#CAD1D8")
WHITE = colors.white


def register_fonts() -> None:
    if not FONT_PATH.exists():
        raise FileNotFoundError(f"Chinese font not found: {FONT_PATH}")
    pdfmetrics.registerFont(TTFont(FONT_NAME, str(FONT_PATH)))


def build_styles():
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "TitleCN",
            parent=styles["Title"],
            fontName=FONT_NAME,
            fontSize=21,
            leading=28,
            textColor=NAVY,
            alignment=TA_CENTER,
            spaceAfter=5 * mm,
        ),
        "subtitle": ParagraphStyle(
            "SubtitleCN",
            parent=styles["Normal"],
            fontName=FONT_NAME,
            fontSize=10,
            leading=15,
            textColor=MID_GRAY,
            alignment=TA_CENTER,
            spaceAfter=6 * mm,
        ),
        "h1": ParagraphStyle(
            "HeadingCN",
            parent=styles["Heading1"],
            fontName=FONT_NAME,
            fontSize=14,
            leading=19,
            textColor=NAVY,
            spaceBefore=3 * mm,
            spaceAfter=2.5 * mm,
        ),
        "body": ParagraphStyle(
            "BodyCN",
            parent=styles["BodyText"],
            fontName=FONT_NAME,
            fontSize=9.5,
            leading=15,
            textColor=colors.HexColor("#20252A"),
            alignment=TA_LEFT,
        ),
        "note": ParagraphStyle(
            "NoteCN",
            parent=styles["BodyText"],
            fontName=FONT_NAME,
            fontSize=8.3,
            leading=12,
            textColor=MID_GRAY,
            alignment=TA_LEFT,
        ),
        "callout": ParagraphStyle(
            "CalloutCN",
            parent=styles["BodyText"],
            fontName=FONT_NAME,
            fontSize=10.2,
            leading=16,
            textColor=NAVY,
            borderColor=BLUE,
            borderWidth=1,
            borderPadding=8,
            backColor=LIGHT_BLUE,
            spaceAfter=5 * mm,
        ),
        "cell": ParagraphStyle(
            "CellCN",
            parent=styles["BodyText"],
            fontName=FONT_NAME,
            fontSize=8.7,
            leading=12,
            textColor=colors.HexColor("#20252A"),
            alignment=TA_LEFT,
        ),
        "cell_center": ParagraphStyle(
            "CellCenterCN",
            parent=styles["BodyText"],
            fontName=FONT_NAME,
            fontSize=8.7,
            leading=12,
            textColor=colors.HexColor("#20252A"),
            alignment=TA_CENTER,
        ),
        "header": ParagraphStyle(
            "HeaderCN",
            parent=styles["BodyText"],
            fontName=FONT_NAME,
            fontSize=8.8,
            leading=12,
            textColor=WHITE,
            alignment=TA_CENTER,
        ),
    }


def p(text: object, style) -> Paragraph:
    return Paragraph(str(text), style)


def make_table(headers, rows, widths, styles, align_first="LEFT") -> Table:
    formatted = [[p(value, styles["header"]) for value in headers]]
    for row in rows:
        formatted.append(
            [
                p(value, styles["cell"] if index == 0 else styles["cell_center"])
                for index, value in enumerate(row)
            ]
        )
    table = Table(formatted, colWidths=widths, repeatRows=1, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("GRID", (0, 0), (-1, -1), 0.45, GRID),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("ALIGN", (0, 1), (0, -1), align_first),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]
    for row_index in range(1, len(formatted)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), LIGHT_GRAY))
    table.setStyle(TableStyle(commands))
    return table


def add_section(story, title, table, styles, note=None):
    items = [p(title, styles["h1"]), table]
    if note:
        items.extend([Spacer(1, 1.5 * mm), p(note, styles["note"])])
    items.append(Spacer(1, 3.5 * mm))
    story.append(KeepTogether(items))


def draw_page(canvas, doc):
    canvas.saveState()
    width, height = A4
    canvas.setStrokeColor(GRID)
    canvas.setLineWidth(0.5)
    canvas.line(18 * mm, 14 * mm, width - 18 * mm, 14 * mm)
    canvas.setFont(FONT_NAME, 7.5)
    canvas.setFillColor(MID_GRAY)
    canvas.drawString(18 * mm, 9.5 * mm, "R5 / Wong-Wang 改进模型 - 代表性子样本结果")
    canvas.drawRightString(width - 18 * mm, 9.5 * mm, f"第 {doc.page} 页")
    canvas.restoreState()


def build_pdf() -> None:
    register_fonts()
    styles = build_styles()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=20 * mm,
        title="改进后的R5 / Wong-Wang模型：结果表与局限",
        author="VAM-studying project",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="normal")
    doc.addPageTemplates(PageTemplate(id="main", frames=frame, onPage=draw_page))

    story = [
        p("改进后的R5 / Wong-Wang模型", styles["title"]),
        p("当前能力、样本构成、拟合结果与解释边界", styles["subtitle"]),
        p(
            "当前模型已经能够在choice与RT来自同一决策时刻的前提下，使所有试次真正达到决策阈值，并在当前代表性子样本上较好重现两个年龄组的准确率、RT分位数和不一致CAF。",
            styles["callout"],
        ),
    ]

    add_section(
        story,
        "1. 样本构成",
        make_table(
            ["年龄组", "被试数", "试次数", "一致试次", "不一致试次"],
            [
                ["年轻组 20-29岁", "12", "5,000", "2,502", "2,498"],
                ["年长组 80-89岁", "4", "5,000", "2,509", "2,491"],
                ["合计", "16", "10,000", "5,011", "4,989"],
            ],
            [45 * mm, 25 * mm, 31 * mm, 34 * mm, 34 * mm],
            styles,
        ),
        styles,
        "注意：10,000个trial不是10,000个独立样本。年轻组每位被试贡献175-1,297个试次，年长组每位被试贡献1,015-1,369个试次。",
    )

    add_section(
        story,
        "2. 决策完成率",
        make_table(
            ["年龄组", "达到阈值比例", "未越界试次"],
            [["年轻组", "100%", "0"], ["年长组", "100%", "0"]],
            [70 * mm, 50 * mm, 49 * mm],
            styles,
        ),
        styles,
        "所有预测RT都来自实际越界时刻加非决策时间，没有把模拟截止值当作真实RT。",
    )

    add_section(
        story,
        "3. 总体准确率",
        make_table(
            ["年龄组", "人类准确率", "模型准确率", "绝对差异"],
            [
                ["年轻组", "94.9%", "96.1%", "+1.1个百分点"],
                ["年长组", "97.6%", "97.9%", "+0.4个百分点"],
            ],
            [48 * mm, 40 * mm, 40 * mm, 41 * mm],
            styles,
        ),
        styles,
    )

    add_section(
        story,
        "4. 不一致条件准确率",
        make_table(
            ["年龄组", "人类", "模型", "绝对差异"],
            [
                ["年轻组", "91.7%", "92.2%", "+0.5个百分点"],
                ["年长组", "96.1%", "95.9%", "-0.2个百分点"],
            ],
            [48 * mm, 40 * mm, 40 * mm, 41 * mm],
            styles,
        ),
        styles,
        "两组中较大的不一致准确率差距低于0.5个百分点。",
    )

    story.append(PageBreak())

    add_section(
        story,
        "5. 反应时间与分布拟合",
        make_table(
            ["年龄组", "人类平均RT", "模型平均RT", "RT分位数MAE"],
            [
                ["年轻组", "0.603秒", "0.592秒", "0.061秒"],
                ["年长组", "0.941秒", "0.891秒", "0.038秒"],
            ],
            [48 * mm, 40 * mm, 40 * mm, 41 * mm],
            styles,
        ),
        styles,
        "模型重现了年长组慢于年轻组的方向，但年长组平均RT仍偏快约0.050秒。",
    )

    add_section(
        story,
        "6. 不一致条件CAF",
        make_table(
            ["年龄组", "不一致CAF RMSE", "主要结果"],
            [
                ["年轻组", "0.028", "重现随RT增加而提高的准确率"],
                ["年长组", "0.009", "与人类CAF整体非常接近"],
            ],
            [44 * mm, 42 * mm, 83 * mm],
            styles,
        ),
        styles,
    )

    add_section(
        story,
        "7. 一致条件准确率",
        make_table(
            ["年龄组", "人类", "模型", "当前问题"],
            [
                ["年轻组", "98.2%", "100%", "模型没有产生一致错误"],
                ["年长组", "99.0%", "100%", "模型没有产生一致错误"],
            ],
            [42 * mm, 32 * mm, 32 * mm, 63 * mm],
            styles,
        ),
        styles,
    )

    add_section(
        story,
        "8. 快速错误效应",
        make_table(
            ["年龄组", "人类：错误RT-正确RT", "模型：错误RT-正确RT", "判断"],
            [
                ["年轻组", "-0.064秒", "-0.151秒", "模型快速错误过强"],
                ["年长组", "-0.210秒", "-0.242秒", "方向和幅度较接近"],
            ],
            [36 * mm, 51 * mm, 51 * mm, 31 * mm],
            styles,
        ),
        styles,
        "负值表示错误反应快于正确反应。",
    )

    add_section(
        story,
        "9. 内部机制对应关系",
        make_table(
            ["年龄组", "输入反转", "WW状态反转", "读出前target恢复", "正确试次中恢复"],
            [
                ["年轻组", "0.077秒", "0.139秒", "86.0%", "93.4%"],
                ["年长组", "0.116秒", "0.217秒", "95.8%", "99.9%"],
            ],
            [37 * mm, 32 * mm, 35 * mm, 36 * mm, 35 * mm],
            styles,
        ),
        styles,
        "两个年龄组的错误试次在读出前target恢复率均为0%，因此错误发生在target恢复之前，而不是由决策后的信息改写。",
    )

    story.append(PageBreak())

    add_section(
        story,
        "10. 当前模型能够实现的能力",
        make_table(
            ["能力", "当前状态"],
            [
                ["Choice-RT一致性", "在持续越界时刻同时决定choice与RT，忽略后续信息"],
                ["真实决策完成", "两组共10,000个试次均真实越界"],
                ["行为拟合", "总体准确率、不一致准确率、RT分位数和不一致CAF均较接近人类"],
                ["年龄差异", "重现年长组总体慢于年轻组"],
                ["机制方向", "正确反应通常发生在target恢复后，错误发生在恢复前"],
            ],
            [50 * mm, 119 * mm],
            styles,
        ),
        styles,
    )

    add_section(
        story,
        "11. 当前局限",
        make_table(
            ["局限", "具体含义"],
            [
                ["被试少且不平衡", "年轻组12人、年长组4人；大量trial不能替代独立被试数量"],
                ["尚无独立验证", "时间安排在当前10,000个试次上选择，未在留出被试或刺激上检验"],
                ["年龄组使用不同时间设置", "差异可能反映真实年龄机制，也可能是小样本适配"],
                ["缺少一致错误", "尚未模拟偶发注意失误、感觉噪声或按键错误"],
                ["RT分布仍有偏差", "条件间分离偏大，模型长尾短于人类，尤其缺少年长组极慢反应"],
                ["年轻组快速错误过强", "模型错误与正确RT差约-0.151秒，人类约-0.064秒"],
                ["机制尚未被验证", "结果证明计算可行性，不能证明人脑采用相同时间映射或冲突机制"],
                ["当前是组水平诊断", "尚未建立可靠的个体参数模型，也没有完整全样本拟合"],
            ],
            [55 * mm, 114 * mm],
            styles,
        ),
        styles,
    )

    story.extend(
        [
            p("12. 结论边界", styles["h1"]),
            p(
                "当前模型通过了代表性小样本上的机制可行性检验：在choice与RT严格同步、全部试次真实越界的条件下，能够较好重现两个年龄组的准确率、RT分位数和不一致CAF。但由于被试少、年龄组不平衡、没有独立验证且完整RT分布仍有偏差，目前不能将其视为完成的年龄机制模型或全样本行为模型。",
                styles["callout"],
            ),
            Spacer(1, 3 * mm),
            p(
                "数据来源：artifacts/results/r5_choice_coupled_schedule_optimization_20260803/selected_trial_level_predictions.csv。当前结果为探索性代表性子样本优化。",
                styles["note"],
            ),
        ]
    )

    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build_pdf()
