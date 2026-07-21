const path = require('path');
const ROOT = path.resolve(__dirname, '..');
const ASSETS = path.join(ROOT, 'poster', '.build', 'assets');
const pptxgen = require('pptxgenjs');

const pptx = new pptxgen();
pptx.defineLayout({ name: 'A1_PORTRAIT', width: 23.39, height: 33.11 });
pptx.layout = 'A1_PORTRAIT';
pptx.author = 'Vladislav Lapin';
pptx.subject = 'AIRI Summer School 2026 poster: DoRA';
pptx.title = 'Когда DoRA помогает?';
pptx.company = 'MIPT FPMI / AI360';
pptx.lang = 'ru-RU';
pptx.theme = { headFontFace: 'Arial', bodyFontFace: 'Arial', lang: 'ru-RU' };
pptx.margin = 0;

const C = {
  ink: '15242D', muted: '66747C', grid: 'DDE5E7', pale: 'F3F7F7',
  pale2: 'EAF5F3', teal: '0B8F83', white: 'FFFFFF', lightText: '8A959B'
};

const slide = pptx.addSlide();
slide.background = { color: C.white };

const addText = (text, x, y, w, h, options={}) => {
  slide.addText(text, {
    x, y, w, h, fontFace: 'Arial', fontSize: 17, color: C.ink,
    margin: 0, breakLine: false, valign: 'top', fit: 'shrink', ...options,
  });
};

const sectionTitle = (text, x, y, w) => {
  addText(text, x, y, w, 0.48, { fontSize: 25, color: C.ink });
  slide.addShape(pptx.ShapeType.line, {
    x, y: y + 0.54, w: 0.95, h: 0, line: { color: C.teal, width: 3.2 }
  });
};

const bullet = (label, body, x, y, w, h=0.65, fontSize=16.4, color=C.ink) => {
  slide.addText([
    { text: '• ', options: { bold: true, color: C.teal } },
    { text: label, options: { bold: true, color } },
    { text: body, options: { color } },
  ], { x, y, w, h, fontFace:'Arial', fontSize, color, margin:0, valign:'top', fit:'shrink' });
};

slide.addImage({ path: path.join(ASSETS, 'header_logos.png'), x: 1.0, y: 0.42, w: 8.4, h: 1.73 });
addText('ЛЕТО С AIRI 2026', 17.35, 0.62, 5.0, 0.55, { fontSize: 22, bold: true, align: 'right', color: '111111' });
addText('Когда DoRA помогает?', 2.0, 2.15, 19.39, 0.82, { fontSize: 43, align: 'center', color: '111111' });
addText('Геометрия адаптеров, бюджет параметров и few-shot-адаптация на отложенных сидах', 1.45, 3.05, 20.49, 0.92, { fontSize: 26, align: 'center' });
addText('Владислав Лапин  ·  МФТИ ФПМИ / AI360  ·  Лето с AIRI 2026', 3.4, 4.02, 16.6, 0.38, { fontSize: 15.5, align: 'center', color: C.muted });

slide.addShape(pptx.ShapeType.line, { x: 11.70, y: 5.02, w: 0, h: 24.78, line: { color: C.grid, width: 1.3 } });
const LX = 1.05, LW = 10.05, RX = 12.28, RW = 10.05;

sectionTitle('Мотивация и исследовательский вопрос', LX, 5.04, LW);
addText('LoRA задаёт низкоранговое обновление весов, но одновременно меняет направление строки и её норму. DoRA разделяет эти две степени свободы.', LX, 5.78, LW, 1.12, { fontSize: 17.0 });
slide.addText([
  { text: 'Вопрос: ', options: { bold: true } },
  { text: 'помогает ли это при целевом сдвиге после честного выбора конфигурации, контроля числа параметров и негативных контролей?' }
], { x:LX, y:6.93, w:LW, h:1.18, fontFace:'Arial', fontSize:17.0, color:C.ink, margin:0, valign:'top', fit:'shrink' });

sectionTitle('Норма и направление', LX, 8.30, LW);
const by = 9.18;
addText('LoRA', LX, by+0.36, 1.05, 0.34, { fontSize: 17, bold:true, color:'4C91AE' });
slide.addShape(pptx.ShapeType.roundRect, { x:LX+1.2, y:by, w:1.2, h:0.95, fill:{color:'F5F6F7'}, line:{color:'AAB4B9', width:1.3} });
addText('W₀', LX+1.2, by+0.26, 1.2, 0.35, { fontSize:17, bold:true, align:'center' });
slide.addShape(pptx.ShapeType.chevron, { x:LX+2.55, y:by+0.24, w:0.45, h:0.47, fill:{color:C.grid}, line:{color:C.grid} });
slide.addShape(pptx.ShapeType.roundRect, { x:LX+3.0, y:by, w:1.45, h:0.95, fill:{color:'FFF5DD'}, line:{color:'D7A425', width:1.3} });
addText('+ BA', LX+3.0, by+0.25, 1.45, 0.35, { fontSize:17, bold:true, align:'center' });
slide.addShape(pptx.ShapeType.chevron, { x:LX+4.62, y:by+0.24, w:0.45, h:0.47, fill:{color:C.grid}, line:{color:C.grid} });
slide.addShape(pptx.ShapeType.roundRect, { x:LX+5.07, y:by, w:2.65, h:0.95, fill:{color:'EAF3F8'}, line:{color:'4C91AE', width:1.3} });
addText('Ŵ = W₀ + BA', LX+5.07, by+0.25, 2.65, 0.35, { fontSize:16.2, bold:true, align:'center' });
addText('направление и норма связаны', LX+5.15, by+1.02, 2.55, 0.32, { fontSize:12.5, align:'center', color:C.muted });

const dy = by + 1.65;
addText('DoRA', LX, dy+0.36, 1.05, 0.34, { fontSize: 17, bold:true, color:C.teal });
slide.addShape(pptx.ShapeType.roundRect, { x:LX+1.2, y:dy, w:1.2, h:0.95, fill:{color:'F5F6F7'}, line:{color:'AAB4B9', width:1.3} });
addText('W₀', LX+1.2, dy+0.26, 1.2, 0.35, { fontSize:17, bold:true, align:'center' });
slide.addShape(pptx.ShapeType.chevron, { x:LX+2.55, y:dy+0.24, w:0.45, h:0.47, fill:{color:C.grid}, line:{color:C.grid} });
slide.addShape(pptx.ShapeType.roundRect, { x:LX+3.0, y:dy, w:1.72, h:0.95, fill:{color:'FFF5DD'}, line:{color:'D7A425', width:1.3} });
addText('V = W₀ + BA', LX+3.0, dy+0.25, 1.72, 0.35, { fontSize:15.4, bold:true, align:'center' });
slide.addShape(pptx.ShapeType.chevron, { x:LX+4.88, y:dy+0.24, w:0.45, h:0.47, fill:{color:C.grid}, line:{color:C.grid} });
slide.addShape(pptx.ShapeType.roundRect, { x:LX+5.33, y:dy, w:1.55, h:0.95, fill:{color:'E9F8F6'}, line:{color:C.teal, width:1.3} });
addText('V / ||V||₂', LX+5.33, dy+0.25, 1.55, 0.35, { fontSize:15.4, bold:true, align:'center' });
slide.addShape(pptx.ShapeType.chevron, { x:LX+7.04, y:dy+0.24, w:0.45, h:0.47, fill:{color:C.grid}, line:{color:C.grid} });
slide.addShape(pptx.ShapeType.roundRect, { x:LX+7.49, y:dy, w:1.85, h:0.95, fill:{color:'E9F8F6'}, line:{color:C.teal, width:1.3} });
addText('m ⊙ V/||V||₂', LX+7.49, dy+0.24, 1.85, 0.38, { fontSize:14.4, bold:true, align:'center', color:C.teal });
addText('направление', LX+5.33, dy+1.02, 1.55, 0.31, { fontSize:12.3, align:'center', color:C.muted });
addText('направление и норма разделены', LX+7.26, dy+1.02, 2.30, 0.40, { fontSize:12.1, align:'center', color:C.teal, bold:true });
addText('Ранг r одинаковый; DoRA добавляет d_out обучаемых параметров нормы сверх LoRA.', LX+1.2, dy+1.46, 8.5, 0.42, { fontSize:12.8, align:'center', color:C.muted });
addText('BA управляет направлением, а m — нормой строки.', LX, 12.68, LW, 0.52, { fontSize:16.6, italic:true, align:'center' });

sectionTitle('Протокол зафиксирован до финальной оценки', LX, 13.55, LW);
bullet('Валидация: ', '5 пилотных сидов; скорость обучения, ранняя остановка и распределение ранга.', LX, 14.28, LW, 0.70);
bullet('Финальная оценка: ', 'MLP — 20, CNN — 10 новых отложенных сидов; один фиксированный чекпоинт на архитектуру.', LX, 15.02, LW, 0.83);
bullet('Дизайн: ', 'три сдвига; отдельные семейства сравнений с поправкой Холма m=9/6/3.', LX, 15.90, LW, 0.72);
bullet('Базовые варианты: ', 'замороженная модель, адаптация только величины, LoRA, LoRA+, согласованные по бюджету варианты и полное дообучение.', LX, 16.67, LW, 1.03);
bullet('Устойчивость: ', '4 объёма данных; 10 синтетических целей × 5 инициализаций; 1 960 записей и 1 840 задач обучения/оптимизации.', LX, 17.76, LW, 0.92);

sectionTitle('Негативные контроли ограничивают вывод', LX, 19.00, LW);
bullet('Контраст (MLP): ', 'адаптация только величины 96.90% > DoRA 95.18%.', LX, 19.75, LW, 0.65);
bullet('Поворот (MLP): ', 'LoRA с бюджетом параметров DoRA = DoRA = 93.89%.', LX, 20.43, LW, 0.65);
bullet('Комбинированный сдвиг (MLP): ', 'согласование бюджета параметров уменьшает разрыв до +0.53 п.п.; доверительный интервал включает ноль.', LX, 21.11, LW, 0.88);

slide.addShape(pptx.ShapeType.roundRect, { x:LX, y:22.20, w:LW, h:5.28, fill:{color:C.pale}, line:{color:C.grid, width:1.0} });
addText('Ограничения и границы вывода', LX+0.35, 22.50, LW-0.7, 0.50, { fontSize:22.5, bold:true });
bullet('', 'Digits; один фиксированный предобученный чекпоинт на архитектуру.', LX+0.35, 23.20, LW-0.7, 0.62, 16.0);
bullet('', 'Одно целевое разбиение уже использовалось на предварительном исследовательском этапе.', LX+0.35, 23.88, LW-0.7, 0.82, 16.0);
bullet('', 'Это контролируемое исследование механизма, а не LLM-scale benchmark.', LX+0.35, 24.76, LW-0.7, 0.67, 16.0);
bullet('', 'Не утверждаем, что DoRA всегда лучше LoRA или что эффект переносится на другие чекпоинты и большие модели.', LX+0.35, 25.48, LW-0.7, 1.10, 16.0);

slide.addShape(pptx.ShapeType.roundRect, { x:RX, y:5.05, w:RW, h:2.82, fill:{color:C.pale2}, line:{color:'CBE7E2', width:1.2} });
addText('КОМБИНИРОВАННЫЙ СДВИГ', RX+0.35, 5.33, RW-0.7, 0.42, { fontSize:23, bold:true, color:C.teal });
addText('+1.06 п.п. MLP  ·  +0.92 п.п. CNN', RX+0.35, 5.84, RW-0.7, 0.58, { fontSize:28, bold:true, color:C.teal });
addText('15/20 пар MLP и 8/10 пар CNN — в пользу DoRA. p с поправкой Холма: 0.382 (MLP) / 0.159 (CNN).', RX+0.35, 6.51, RW-0.7, 0.69, { fontSize:16.1, bold:true });
addText('Внутреннее подтверждение на фиксированных чекпоинтах; не независимая внешняя репликация.', RX+0.35, 7.22, RW-0.7, 0.43, { fontSize:14.8, color:C.muted });

slide.addImage({ path:path.join(ASSETS, 'paired_estimates_ru.png'), x:RX, y:8.15, w:RW, h:4.04 });
slide.addImage({ path:path.join(ASSETS, 'data_sweep_ru.png'), x:RX, y:12.44, w:RW, h:4.96 });

sectionTitle('Что позволяют утверждать проверки', RX, 17.73, RW);
bullet('Комбинированный сдвиг: ', 'DoRA достигает 75.22% (MLP) и 80.53% (CNN), близко к полному дообучению (75.17%, 80.67%) при примерно 12–15% обучаемых параметров.', RX, 18.47, RW, 1.16, 16.2);
bullet('Усиленные базовые варианты: ', 'DoRA − LoRA+ = +1.43 п.п. (Holm p=0.0509); при согласованном бюджете LoRA остаётся +0.53 п.п., но ДИ включает ноль.', RX, 19.70, RW, 1.16, 16.2);
bullet('Синтетическая диагностика (γ=0.8): ', 'DoRA точно выражает цель; 44/50 запусков достигают ошибки <10^-3; средняя ошибка SVD-оракула LoRA — 0.289.', RX, 20.93, RW, 1.15, 16.2);
slide.addShape(pptx.ShapeType.roundRect, { x:RX, y:22.85, w:RW, h:3.58, fill:{color:C.ink}, line:{color:C.ink, width:1.0} });
addText('Вывод', RX+0.38, 23.18, RW-0.76, 0.48, { fontSize:24, bold:true, color:C.white });
addText('DoRA выглядит полезной, когда целевой сдвиг одновременно требует менять направление и неоднородно масштабировать строки. Но это осторожная внутренняя оценка, а не универсальная победа над LoRA.', RX+0.38, 23.84, RW-0.76, 1.72, { fontSize:19.2, color:C.white, valign:'mid' });

slide.addShape(pptx.ShapeType.line, { x:1.05, y:30.20, w:21.28, h:0, line:{color:C.grid, width:1.0} });
addText('[1] Hu et al. LoRA. ICLR 2022.   [2] Liu et al. DoRA. ICML 2024.   [3] Hayou et al. LoRA+. ICML 2024.', 1.05, 30.44, 11.0, 0.62, { fontSize:10.7, color:C.muted });
slide.addImage({ path:path.join(ASSETS, 'sber_logo.png'), x:12.36, y:30.60, w:3.6, h:0.98 });
addText('Генеральный партнёр', 12.38, 31.67, 3.55, 0.28, { fontSize:10.5, align:'center', color:C.lightText });
slide.addImage({ path:path.join(ASSETS, 'avito_logo.png'), x:16.42, y:30.65, w:3.25, h:0.78 });
addText('Партнёр', 16.45, 31.67, 3.15, 0.28, { fontSize:10.5, align:'center', color:C.lightText });
slide.addImage({ path:path.join(ASSETS, 'github_qr.png'), x:20.25, y:30.55, w:1.82, h:1.82, hyperlink:{url:'https://github.com/Vladislav-Lp/dora-sanity-experiment'} });

pptx.writeFile({ fileName: path.join(ROOT, 'poster', 'Lapin_Vladislav_DoRA_AIRI_2026_RU.pptx') });
