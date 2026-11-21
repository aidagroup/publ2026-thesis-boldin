"""Progress deck: same visual style as first-try.pdf (960x540, warm bg, cards).

Usage: uv run --no-project --with reportlab python make_deck.py
Output: second-try.pdf (15 slides)
"""

from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

W, H = 960, 540
M = 48  # page margin
G = 14  # gutter between cards

BG = HexColor("#f5f3ee")
CARD = white
BORDER = HexColor("#e3ddcf")
INK = HexColor("#2b3240")
BODY = HexColor("#3a4150")
SOFT = HexColor("#6b7280")
TAN = HexColor("#a3763c")
BLUE = HexColor("#7e9bc9")
BLUE_BG = HexColor("#eef2f9")
TAN_BG = HexColor("#f7f1e6")

pdfmetrics.registerFont(TTFont("Ar", "/System/Library/Fonts/Supplemental/Arial.ttf"))
pdfmetrics.registerFont(TTFont("ArB", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"))
pdfmetrics.registerFont(TTFont("ArI", "/System/Library/Fonts/Supplemental/Arial Italic.ttf"))


def wrap(text, font, size, width):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if pdfmetrics.stringWidth(t, font, size) <= width:
            cur = t
        else:
            if cur:
                lines.append(cur)
            cur = w_
    if cur:
        lines.append(cur)
    return lines


def par(c, x, y, w, text, font="Ar", size=11.5, leading=15, color=BODY):
    c.setFont(font, size)
    c.setFillColor(color)
    for ln in wrap(text, font, size, w):
        c.drawString(x, y, ln)
        y -= leading
    return y


def card(c, x, y, w, h, fill=CARD, stroke=BORDER):
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(1)
    c.roundRect(x, y, w, h, 10, fill=1, stroke=1)


def rubric(c, text):
    t = c.beginText(M, H - 40)
    t.setFont("ArB", 12.5)
    t.setFillColor(TAN)
    t.setCharSpace(2.5)
    t.textOut(text.upper())
    c.drawText(t)


def headline(c, text, size=27, y=None):
    c.setFont("ArB", size)
    c.setFillColor(INK)
    y = y if y is not None else H - 76
    c.drawString(M, y, text)


def footnote(c, text, y=24):
    c.setFont("ArI", 9.5)
    c.setFillColor(SOFT)
    lines = wrap(text, "ArI", 9.5, W - 2 * M - 40)
    yy = y + (len(lines) - 1) * 11.5
    for ln in lines:
        c.drawString(M, yy, ln)
        yy -= 11.5


def pagenum(c, i):
    c.setFont("Ar", 10)
    c.setFillColor(SOFT)
    c.drawRightString(W - M, 16, str(i))


def new_slide(c, i, rub=None, head=None, head_size=27):
    c.setFillColor(BG)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    if rub:
        rubric(c, rub)
    if head:
        headline(c, head, head_size)
    pagenum(c, i)


def card_block(c, x, y, w, h, title, text, fill=CARD, title_color=INK):
    card(c, x, y, w, h, fill=fill)
    pad = 15
    yy = y + h - pad - 15
    c.setFont("ArB", 13.5)
    c.setFillColor(title_color)
    for ln in wrap(title, "ArB", 13.5, w - 2 * pad)[:2]:
        c.drawString(x + pad, yy, ln)
        yy -= 16
    yy -= 2
    par(c, x + pad, yy, w - 2 * pad, text, size=11, leading=14)


c = canvas.Canvas("second-try.pdf", pagesize=(W, H))
c.setTitle("Прогресс ВКР: Bi-JEPA для двух роботов — ноябрь 2026")
c.setAuthor("Ivan Boldin-Iliukhin")

# ---------------- 1. Title ----------------
c.setFillColor(BG)
c.rect(0, 0, W, H, fill=1, stroke=0)
t = c.beginText(M, H - 150)
t.setFont("ArB", 12)
t.setFillColor(TAN)
t.setCharSpace(2.5)
t.textOut("ВЫПУСКНАЯ КВАЛИФИКАЦИОННАЯ РАБОТА")
c.drawText(t)
c.setFont("ArB", 34)
c.setFillColor(INK)
c.drawString(M, H - 205, "Децентрализованная координация")
c.drawString(M, H - 245, "двух роботов через Bi-JEPA")
c.setFont("Ar", 16)
c.setFillColor(BODY)
c.drawString(M, H - 285, "Два робота SO-ARM101 совместно собирают кубик Рубика")
c.setStrokeColor(TAN)
c.setLineWidth(1.5)
c.line(M, H - 315, M + 120, H - 315)
c.setFont("Ar", 12.5)
c.setFillColor(SOFT)
c.drawString(M, H - 345, "Ivan Boldin-Iliukhin, Danil Belov, Pavel Osinenko • AIDA • 2026")
c.setFont("ArB", 12.5)
c.setFillColor(INK)
c.drawString(M, H - 385, "Прогресс с прошлой презентации — ноябрь 2026")
pagenum(c, 1)
c.showPage()

# ---------------- 2. Тема (как было) ----------------
new_slide(c, 2, "Тема", "Почему выбрана эта тема")
cw = (W - 2 * M - 2 * G) / 3
items = [
    ("Роботам нужно понимать друг друга",
     "В жизни связь между роботами может пропасть. Поэтому каждый робот должен сам догадываться, что собирается сделать его партнёр."),
    ("Задача, где без согласия не обойтись",
     "Чтобы повернуть грань кубика Рубика, одна рука должна его держать, а другая крутить. Если руки не договорятся, кубик сдвинется и сборка собьётся."),
    ("Доступные роботы",
     "Руки SO-ARM101 недорогие и с открытым кодом. Поэтому результат можно потом проверить на настоящих роботах."),
]
for i, (ti, tx) in enumerate(items):
    x = M + i * (cw + G)
    card_block(c, x, 90, cw, 330, ti, tx)
footnote(c, "Как и в прошлой презентации: предмет работы — согласование двух роботов без обмена сообщениями.")
c.showPage()

# ---------------- 3. Цель (как было) ----------------
new_slide(c, 3, "Цель")
c.setFont("ArB", 24)
c.setFillColor(INK)
c.drawString(M, H - 130, "Разработать и проверить JEPA-подход к децентрализованной")
c.drawString(M, H - 165, "координации двух роботов-манипуляторов на задаче")
c.drawString(M, H - 200, "совместной сборки кубика Рубика")
c.setFont("Ar", 15)
c.setFillColor(BODY)
c.drawString(M, H - 240, "и показать его преимущество над подходами, которые моделируют партнёра через")
c.drawString(M, H - 262, "реконструкцию или дистилляцию")
footnote(c, "JEPA — архитектура, которая предсказывает не саму картинку, а её сжатое описание. Латент — это и есть такое сжатое описание ситуации, которое модель выучивает из наблюдений.")
c.showPage()

# ---------------- 4. Постановка задачи (как было) ----------------
new_slide(c, 4, "Постановка задачи", "Сборка кубика по готовому плану")
steps = [
    ("Солвер", "математический алгоритм заранее считает порядок ходов, около 20"),
    ("План", "обе руки знают, какую грань и на сколько градусов повернуть"),
    ("Согласование", "руки сами решают, кто держит, кто крутит и когда отпустить"),
    ("Ход сделан", "грань повёрнута, остальной кубик на месте"),
]
sw = (W - 2 * M - 3 * G) / 4
for i, (ti, tx) in enumerate(steps):
    x = M + i * (sw + G)
    fill = TAN_BG if i == 2 else CARD
    card_block(c, x, 300, sw, 150, ti, tx, fill=fill, title_color=TAN if i == 2 else INK)
    if i < 3:
        c.setFont("ArB", 14)
        c.setFillColor(SOFT)
        c.drawString(x + sw + G / 2 - 4, 368, "→")
bullets = [
    "Планировать ходы роботам не нужно — это делает классический алгоритм",
    "Учатся они другому: слаженно выполнять каждый ход без обмена сообщениями",
    "Поворот одной грани, как в видео в конце, — это один такой ход",
]
yy = 270
for b in bullets:
    c.setFont("ArB", 11.5)
    c.setFillColor(TAN)
    c.drawString(M, yy, "•")
    yy = par(c, M + 16, yy, W - 2 * M - 16, b, size=11.5, leading=14) - 8
footnote(c, "Солвер — например, алгоритм Коциембы: находит короткую последовательность поворотов граней, которая собирает кубик. Сам он не обучается.")
c.showPage()

# ---------------- 5. Что изменилось (новый) ----------------
new_slide(c, 5, "Что изменилось", "Перешли от непрерывных команд к готовым приёмам")
cards5 = [
    ("Проверили на сервере",
     "Прогнали обучение с подражанием: 364 обучающих эпизода из 512. Холдер научился держать куб всегда — 100%. Ротатор действует верно лишь в 2% случаев. Итог — 0 собранных кубиков.", CARD, INK),
    ("Почему не вышло",
     "Ротатор не видит челюсть холдера в своих наблюдениях. На уровне непрерывных команд это почти не лечится: рука “не понимает”, где чужой захват.", CARD, INK),
    ("Что решили",
     "Руки теперь выбирают готовые приёмы: подойти, взять, повернуть, отпустить. Обучается только выбор и очерёдность приёмов. Движения внутри приёма заданы заранее и открыто записаны в допущения работы.", BLUE_BG, INK),
]
for i, (ti, tx, fl, tc) in enumerate(cards5):
    x = M + i * (cw + G)
    card_block(c, x, 90, cw, 330, ti, tx, fill=fl, title_color=tc)
footnote(c, "Приём (примитив) — готовое движение руки с началом и концом. Раньше руки обучались каждой мелкой команде, теперь — только тому, какой приём выполнить следующим.")
c.showPage()

# ---------------- 6. Рамка экспериментов (новый) ----------------
new_slide(c, 6, "Рамка экспериментов", "Лестница задач и три режима видимости партнёра")
lw = (W - 2 * M - G) * 0.52
c.setFont("ArB", 13)
c.setFillColor(INK)
c.drawString(M, H - 116, "Ступени задачи")
ladder = [
    ("E1 — один ход", "повернуть одну грань; проверка, что обучение работает", CARD),
    ("E2 — кубик, 2–3 хода", "с переворотами куба; цель к предзащите 14 декабря", TAN_BG),
    ("E3 — случайный план", "5 и больше ходов, порядок не фиксирован", CARD),
    ("E4 — полная сборка", "план строит солвер Коциембы поверх E3", CARD),
]
yy = H - 140
for ti, tx, fl in ladder:
    hh = 62
    card(c, M, yy - hh, lw, hh, fill=fl)
    c.setFont("ArB", 11.5)
    c.setFillColor(INK)
    c.drawString(M + 14, yy - 20, ti)
    par(c, M + 14, yy - 37, lw - 28, tx, size=10.5, leading=12.5)
    yy -= hh + 10
rx = M + lw + G
rw = W - M - rx
c.setFont("ArB", 13)
c.setFillColor(INK)
c.drawString(rx, H - 116, "Что видит робот о партнёре")
modes = [
    ("Партнёр не виден", "никакой информации о другой руке — нижняя граница"),
    ("Партнёр известен", "полная информация — верхняя граница"),
    ("Партнёр предсказан", "Bi-JEPA угадывает состояние партнёра — весна, главный вопрос работы"),
]
yy = H - 140
for ti, tx in modes:
    hh = 62
    card(c, rx, yy - hh, rw, hh, fill=BLUE_BG if "предсказан" in ti else CARD)
    c.setFont("ArB", 11.5)
    c.setFillColor(INK)
    c.drawString(rx + 14, yy - 20, ti)
    par(c, rx + 14, yy - 37, rw - 28, tx, size=10.5, leading=12.5)
    yy -= hh + 10
c.setFont("ArB", 13)
c.setFillColor(INK)
c.drawString(rx, yy - 6, "Как сравниваем")
par(c, rx, yy - 26, rw, "Успех в парах роботов из разных обучений, разница между режимами видимости, восстановление после сбоев.", size=10.5, leading=12.5)
footnote(c, "Cross-play: двое рук, обученных порознь, встречаются впервые. Если сборка всё равно выходит — согласование настоящее, а не заученное.")
c.showPage()

# ---------------- 7. Новая среда (новый) ----------------
new_slide(c, 7, "Новая среда", "Многослойный куб и приёмы вместо команд")
cards7 = [
    ("Кубик",
     "Ядро и шесть поворотных слоёв. Слои вращаются на 90° и 180°. Физика мешает повернуть слой, который зажат другой рукой."),
    ("Приёмы и решения",
     "Каждые несколько шагов обе руки одновременно выбирают по приёму. В среду встроены сбои: захват может не удаться, грань — проскользнуть."),
    ("Дирижёр и проверки",
     "Скриптовый эталон прошёл всю цепочку хода: переворот куба, подход, захват, поворот, отпускание. Среда проверена 337 автотестами. Один ход считается в 6 раз быстрее: 0.34 с вместо 1.9 с."),
]
for i, (ti, tx) in enumerate(cards7):
    x = M + i * (cw + G)
    card_block(c, x, 90, cw, 330, ti, tx)
footnote(c, "Дирижёр — заранее записанный “идеальный” сценарий. Он показывает, что в среде собрать кубик реально. Обучение на нём не строится.")
c.showPage()

# ---------------- 8. Робот (как было) ----------------
new_slide(c, 8, "Робот", "SO-ARM101 с параллельным захватом")
nums = [
    ("5+1", "суставов у каждой руки: пять подвижных и привод захвата"),
    ("2", "руки в сцене: холдер держит, ротатор крутит"),
    ("84 мм", "между основаниями рук в симуляции"),
    ("90°", "поворот грани за один ход"),
]
nw = (W - 2 * M - 3 * G) / 4
for i, (big, cap) in enumerate(nums):
    x = M + i * (nw + G)
    card(c, x, 210, nw, 190)
    c.setFont("ArB", 30)
    c.setFillColor(TAN)
    c.drawCentredString(x + nw / 2, 352, big)
    par(c, x + 16, 322, nw - 32, cap, size=11, leading=14)
par(c, M, 170, W - 2 * M,
    "Открытая и недорогая рука из проекта LeRobot, на захвате стоит камера. Роли фиксированы: холдер держит корпус кубика, ротатор поворачивает грань.",
    size=12, leading=15, color=BODY)
footnote(c, "Параллельный захват: две губки сходятся навстречу и остаются параллельными, как тиски — грань зажимается плоско.")
c.showPage()

# ---------------- 9. Смена захвата (как было) ----------------
new_slide(c, 9, "Смена робота", "Почему первый захват не подошёл")
c2 = (W - 2 * M - G) / 2
card_block(c, M, 110, c2, 310, "Встроенный SO-100",
           "• Одна подвижная губка, которая поворачивается на шарнире, как клешня\n• Губка движется по дуге и подходит к грани под углом\n• Грань кубика зажимается плохо")
card_block(c, M + c2 + G, 110, c2, 310, "SO-ARM101 + Robonine",
           "• Две губки, которые сходятся по прямой и остаются параллельными\n• Грань кубика зажимается плоско, как в тисках\n• Управление простое: 5 суставов руки и одно движение захвата", fill=TAN_BG)
footnote(c, "Цена замены: пришлось собрать собственную модель робота в ManiSkill и упростить форму губок для расчёта столкновений, иначе зазор между ними “заполнялся” и захват не мог взять кубик.")
c.showPage()

# ---------------- 10. Литература (как было) ----------------
new_slide(c, 10, "Обзор литературы", "Как соседние работы моделируют партнёра")
rows = [
    ("Distilling Collaborative Dynamics into Latent Space…, 2026", "Дистилляция, демонстрации", "Нет"),
    ("Dreaming Of Others: Latent Teammate Modeling…, 2026", "Реконструкция, RL", "Нет опытов"),
    ("Agent Modeling as Auxiliary Task…, 2019", "Предсказание действий, RL", "Нет"),
    ("Theory of Mind as Intrinsic Motivation…, 2023", "Предсказание действий и целей, RL", "Нет"),
    ("Эта работа: Bi-JEPA, предсказание латента партнёра", "Без реконструкции, RL", "Цель: SO-ARM101"),
]
col_w = [W - 2 * M - 300 - 220, 300, 220]
x0 = [M, M + col_w[0], M + col_w[0] + col_w[1]]
yTop = H - 128
rh = 46
c.setFont("ArB", 11.5)
c.setFillColor(SOFT)
c.drawString(x0[0], yTop + 12, "Работа")
c.drawString(x0[1], yTop + 12, "Партнёр и обучение")
c.drawString(x0[2], yTop + 12, "Реальный робот")
yy = yTop
for i, row in enumerate(rows):
    last = i == len(rows) - 1
    fill = BLUE_BG if last else (CARD if i % 2 == 0 else HexColor("#faf8f2"))
    c.setFillColor(fill)
    c.setStrokeColor(BORDER)
    c.rect(M, yy - rh, W - 2 * M, rh, fill=1, stroke=1)
    c.setFont("ArB" if last else "Ar", 10.5)
    c.setFillColor(INK)
    c.drawString(x0[0] + 10, yy - rh + 18, row[0][:70])
    c.setFont("Ar", 10.5)
    c.drawString(x0[1] + 10, yy - rh + 18, row[1])
    c.drawString(x0[2] + 10, yy - rh + 18, row[2])
    yy -= rh
footnote(c, "Координация без связи уже изучена. Свободна JEPA-постановка: прямых аналогов не найдено.")
c.showPage()

# ---------------- 11. Архитектура (как было + строка) ----------------
new_slide(c, 11, "Модель и архитектура", "Bi-JEPA: каждый агент предсказывает партнёра")
flow = [
    ("Наблюдение", "только своё"),
    ("Энкодер", "латент агента"),
    ("Предсказатель", "будущий латент партнёра"),
    ("Политика", "действие руки"),
]
fw = (W - 2 * M - 3 * G) / 4
for i, (ti, tx) in enumerate(flow):
    x = M + i * (fw + G)
    card(c, x, 330, fw, 110)
    c.setFont("ArB", 12.5)
    c.setFillColor(INK)
    c.drawCentredString(x + fw / 2, 408, ti)
    par(c, x + 12, 385, fw - 24, tx, size=11, leading=13.5)
    if i < 3:
        c.setFont("ArB", 14)
        c.setFillColor(SOFT)
        c.drawString(x + fw + G / 2 - 4, 378, "→")
bullets11 = [
    "Ошибка предсказателя считается в латентном пространстве, цель задаёт EMA-копия энкодера",
    "Обучение: RL (IPPO и MAPPO) плюс вспомогательный JEPA-лосс",
    "На исполнении нет обмена сообщениями и центрального контроллера",
    "Весной модуль подключается поверх новой линии: предсказание партнёра помогает выбирать приёмы (режим “партнёр предсказан”)",
]
yy = 300
for b in bullets11:
    c.setFont("ArB", 11.5)
    c.setFillColor(TAN)
    c.drawString(M, yy, "•")
    yy = par(c, M + 16, yy, W - 2 * M - 16, b, size=11.5, leading=14) - 7
footnote(c, "EMA — медленно обновляемая копия энкодера, она даёт стабильную цель для обучения.")
c.showPage()

# ---------------- 12. Особенности (как было) ----------------
new_slide(c, 12, "В чём особенность", "Четыре отличия работы")
cards12 = [
    ("JEPA вместо реконструкции", "Партнёр предсказывается в латенте, без восстановления наблюдений."),
    ("RL, а не имитация", "Соседние работы учатся на демонстрациях, здесь — обучение с подкреплением."),
    ("Тяжёлая контактная задача", "Сборка кубика: на каждом ходу одна рука держит, другая крутит."),
    ("Чистая проверка вклада", "Ablation: партнёр известен, предсказан или не виден."),
]
qw = (W - 2 * M - G) / 2
qh = 130
for i, (ti, tx) in enumerate(cards12):
    x = M + (i % 2) * (qw + G)
    y = 90 + (1 - i // 2) * (qh + G)
    card_block(c, x, y, qw, qh, ti, tx)
footnote(c, "Ablation — сравнение системы с выключенными частями: так видно, какую пользу даёт именно предсказание партнёра.")
c.showPage()

# ---------------- 13. Неудача -> решение (новый) ----------------
new_slide(c, 13, "Главное препятствие", "Грань физически не повернуть — сделали честный обход")
card_block(c, M, 110, c2, 310, "Неделя попыток",
           "Ротатор сжимал грань губками и крутил запястье. Грань почти не двигалась: угол доходил лишь до 2°, куб смещался на 9 сантиметров. Контакт губок с гранью оказался слишком капризным, чтобы успеть в срок.")
card_block(c, M + c2 + G, 110, c2, 310, "Кинематический ассист",
           "Пока ротатор действительно держит грань, среда сама доводит поворот слоя — синхронно с запястьем руки. Без настоящего захвата ассист ничего не делает. Это открыто записано в допущения: движения заданы, предмет работы — согласование.", fill=TAN_BG)
footnote(c, "Ассист не “крутит воздух”: без реального захвата грани поворот не происходит, и ход считается неудавшимся.")
c.showPage()

# ---------------- 14. Демо (новый, слоты) ----------------
new_slide(c, 14, "Демонстрация", "Дирижёр проводит весь ход: от переворота до поворота грани")
steps14 = ["Переворот куба (холдер)", "Подход (ротатор)", "Захват грани", "Поворот грани", "Отпускание"]
swid = (W - 2 * M - 4 * G) / 5
for i, s in enumerate(steps14):
    x = M + i * (swid + G)
    c.setFont("ArB", 10.5)
    c.setFillColor(INK)
    lines = wrap(s, "ArB", 10.5, swid)
    yy = 330
    for ln in lines:
        c.drawCentredString(x + swid / 2, yy, ln)
        yy -= 13
    c.setStrokeColor(BLUE)
    c.setDash(5, 4)
    c.setFillColor(HexColor("#f4f6fa"))
    c.roundRect(x, 150, swid, 120, 8, fill=1, stroke=1)
    c.setDash()
    c.setFont("ArI", 10)
    c.setFillColor(SOFT)
    c.drawCentredString(x + swid / 2, 203, "кадр")
    c.drawCentredString(x + swid / 2, 189, "с сервера")
    if i < 4:
        c.setFont("ArB", 13)
        c.setFillColor(SOFT)
        c.drawString(x + swid + G / 2 - 4, 205, "→")
par(c, M, 115, W - 2 * M,
    "Кадры и видео будут записаны на облачном сервере после его запуска — места под них уже зарезервированы.",
    size=11.5, leading=14, color=BODY)
footnote(c, "Это запись скриптового эталона, а не обученной политики: сначала доказываем, что среда собирается, затем учим руки выбирать приёмы.")
c.showPage()

# ---------------- 15. План работ (обновлённый) ----------------
new_slide(c, 15, "Задачи", "План работ")
cols15 = [
    ("Сделано", TAN_BG, [
        "Новая среда: многослойный куб и приёмы",
        "Тренер IPPO с выбором приёмов",
        "Оценка: успех, время, cross-play, сбои",
        "Дирижёр прошёл весь ход, среда проверена",
        "Облачный сервер с восстановлением после обрывов",
    ]),
    ("В работе", BLUE_BG, [
        "Короткая проверка тренера на сервере",
        "Первое обучение E1: партнёр не виден / известен, 3+ запуска",
    ]),
    ("Впереди", CARD, [
        "E2 — кубик 2–3 хода к предзащите 14 декабря",
        "Метрики согласования и видео",
        "Весной — модуль Bi-JEPA (партнёр предсказан) и сравнение с обучением по демонстрациям",
    ]),
]
for i, (ti, fl, items15) in enumerate(cols15):
    x = M + i * (cw + G)
    card(c, x, 90, cw, 330, fill=fl)
    c.setFont("ArB", 14)
    c.setFillColor(INK)
    c.drawString(x + 15, 392, ti)
    yy = 366
    for it in items15:
        c.setFont("ArB", 10.5)
        c.setFillColor(TAN)
        c.drawString(x + 15, yy, "•")
        nxt = par(c, x + 29, yy, cw - 44, it, size=10.5, leading=13)
        yy = nxt - 7
footnote(c, "IPPO — каждый робот учится независимо алгоритмом PPO. MAPPO — тоже PPO, но во время обучения критик видит обоих роботов и кубик.")
c.showPage()

c.save()
import os
print("OK: second-try.pdf,", 15, "slides,", os.path.getsize("second-try.pdf"), "bytes")
