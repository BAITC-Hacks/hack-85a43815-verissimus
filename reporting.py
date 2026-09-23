"""Self-contained forecast reports and independently evaluated snapshot checks.

The report has no JavaScript, network requests, or secrets. All source strings are
HTML escaped; an SVG chart is generated only from finite numeric predictions.
"""

from datetime import datetime, timedelta, timezone
from html import escape
import json
import math
from zoneinfo import ZoneInfo


def _stamp(value):
    """Only accept explicit timezone metadata; do not silently assume UTC."""
    try:
        if not isinstance(value, str):
            return None
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return stamp.astimezone(timezone.utc) if stamp.utcoffset() is not None else None
    except (ValueError, TypeError, OverflowError):
        return None


def _numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check(key, label, status, detail):
    return {"id": key, "label": label, "status": status, "detail": detail}


def _mapping(value):
    return value if isinstance(value, dict) else {}


def forecast_checks(result):
    """Check the saved result, without trusting its success flag or summary.

    pass means the stated condition was checked, not that forecast skill or the
    historical first publication time of the weather is proven.
    """
    result = _mapping(result)
    points = result.get("forecast_sample") or []
    if not isinstance(points, list):
        points = []
    rows = [row if isinstance(row, dict) else {} for row in points]
    horizon = result.get("horizon_hours")
    count_ok = type(horizon) is int and horizon in (24, 48) and len(rows) == horizon
    contract_ok = result.get("status") == "success" and result.get("power_unit") == "normalized"
    checks = [_check("snapshot_contract", "Успешный расчёт в нормализованной шкале",
                     "pass" if contract_ok else "fail", "status=success; power_unit=normalized."
                     if contract_ok else "Статус расчёта или безразмерная шкала не подтверждены.")]
    checks.append(_check("horizon", "Полнота горизонта", "pass" if count_ok else "fail",
                         f"Получено {len(rows)} строк; запрошено {horizon} ч."))
    times = [_stamp(row.get("time")) for row in rows]
    valid_times = bool(times) and all(stamp is not None for stamp in times)
    unique = valid_times and len(set(times)) == len(times)
    checks.append(_check("unique_hours", "Временные метки без повторов", "pass" if unique else "fail",
                         "Все метки имеют UTC-смещение и уникальны." if unique else
                         "Есть повторы, некорректные метки или отсутствует UTC-смещение."))
    continuous = valid_times and all(b - a == timedelta(hours=1) for a, b in zip(times, times[1:]))
    start = _stamp(result.get("forecast_start"))
    if start is not None:
        continuous = continuous and times[0] == start
    grid_status = "fail" if not continuous else ("unknown" if start is None else "pass")
    checks.append(_check("hourly_grid", "Последовательная почасовая сетка", grid_status,
                         "Начало совпадает с forecast_start; шаг между строками — ровно 1 час." if grid_status == "pass"
                         else "Нет корректного forecast_start." if grid_status == "unknown" else
                         "Сетка не совпадает с началом прогноза или содержит пропуски/нарушение порядка."))
    powers = [row.get("predicted_power") for row in rows]
    bounds = bool(powers) and all(_numeric(power) and 0 <= power <= 1 for power in powers)
    checks.append(_check("power_bounds", "Мощность в исходной шкале [0, 1]", "pass" if bounds else "fail",
                         "Все значения конечны и находятся в [0, 1]." if bounds else
                         "Есть пропуски, нечисловые значения или выход за диапазон."))
    weather_values = bool(rows) and all(_numeric(row.get("wind_speed")) and row["wind_speed"] >= 0
                                      and _numeric(row.get("temperature")) for row in rows)
    checks.append(_check("weather_values", "Погодные признаки заполнены", "pass" if weather_values else "fail",
                         "Ветер и температура конечны; скорость ветра неотрицательна." if weather_values else
                         "В погодных признаках есть пропуски или недопустимые значения."))
    issued = _stamp(result.get("issue_time"))
    decision_ok = issued is not None and valid_times and issued < times[0]
    checks.append(_check("decision_cutoff", "Решение принято до первого целевого часа",
                         "pass" if decision_ok else "fail", "Момент решения строго раньше начала прогноза."
                         if decision_ok else "Не подтвержден момент решения до начала прогноза."))
    weather = _mapping(result.get("weather"))
    available = _stamp(weather.get("available_at"))
    run = _stamp(weather.get("run_time"))
    stamps_exist = issued is not None and available is not None and run is not None
    archive_ok = stamps_exist and run <= available <= issued and run <= issued - timedelta(hours=6)
    archive_status = "unknown" if not stamps_exist else ("pass" if archive_ok else "fail")
    checks.append(_check("weather_cutoff", "Выпуск и метка архива не позже решения", archive_status,
                         "Проверены run_time ≤ available_at ≤ issue_time и запас выпуска ≥ 6 ч."
                         if archive_ok else "Нет достаточных временных метаданных либо нарушен срок доступности."))
    sources = weather.get("sources")
    if isinstance(sources, list) and sources and issued is not None:
        source_times = [_stamp(item.get("last_modified")) if isinstance(item, dict) else None for item in sources]
        source_ok = all(stamp is not None and stamp <= issued for stamp in source_times)
        source_ok = source_ok and available is not None and max(source_times) == available
        source_status = "pass" if source_ok else "fail"
        source_detail = f"Проверено {len(sources)} меток HTTP Last-Modified и их максимум."
    else:
        source_status, source_detail = "unknown", "Полный список меток исходных объектов отсутствует в снимке."
    checks.append(_check("source_cutoff", "Метки исходных погодных объектов", source_status, source_detail))
    model = _mapping(result.get("model"))
    training_end = _stamp(model.get("training_end"))
    training_available = _stamp(model.get("training_available_at"))
    training_known = issued is not None and training_end is not None and training_available is not None
    training_ok = training_known and training_end + timedelta(hours=1) <= training_available <= issued
    checks.append(_check("training_cutoff", "Обучение только на завершённых часах", "unknown" if not training_known
                         else ("pass" if training_ok else "fail"),
                         "Последний обучающий час завершён и доступен не позже момента решения."
                         if training_ok else "Нет достаточных метаданных обучения либо нарушен момент отсечения."))
    calibration = model.get("calibration")
    if isinstance(calibration, dict) and calibration.get("applied"):
        cutoff = _stamp(calibration.get("available_at"))
        fit_end = _stamp(calibration.get("fit_end"))
        known = cutoff is not None and issued is not None and fit_end is not None
        status = "unknown" if not known else ("pass" if fit_end + timedelta(hours=1) <= cutoff <= issued else "fail")
        checks.append(_check("calibration_cutoff", "Данные калибровки доступны к моменту решения", status,
                             "Проверены завершение последнего часа калибровки и его доступность до решения."
                             if status == "pass" else "Нет достаточных метаданных калибровки либо нарушен момент отсечения."))
    return checks


STATUS_LABELS = {"pass": "Проверено", "fail": "Ошибка", "unknown": "Не подтверждено"}


def _h(value):
    return escape(str(value), quote=True)


def _time(value, zone="UTC"):
    stamp = _stamp(value)
    if stamp is None:
        return "не указано / нет UTC-смещения"
    try:
        return stamp.astimezone(ZoneInfo(zone)).isoformat(timespec="minutes")
    except (ValueError, KeyError):
        return stamp.isoformat(timespec="minutes") + " (часовой пояс не распознан)"


def _number(value, digits=4):
    return f"{value:.{digits}f}" if _numeric(value) else "—"


def _pairs(values):
    return "<dl>" + "".join(f"<div><dt>{_h(label)}</dt><dd>{_h(value)}</dd></div>"
                             for label, value in values) + "</dl>"


def _power_svg(points, site_timezone):
    powers = [row.get("predicted_power") for row in points]
    if len(powers) < 2 or not all(_numeric(power) and 0 <= power <= 1 for power in powers):
        return '<p class="notice">График недоступен: недостаточно корректных значений мощности.</p>'
    width, height, left, right, top, bottom = 960, 260, 52, 22, 22, 44
    plot_width, plot_height = width - left - right, height - top - bottom
    coords = [(left + i * plot_width / (len(powers) - 1), top + (1 - value) * plot_height)
              for i, value in enumerate(powers)]
    line = " ".join(f"{x:.2f},{y:.2f}" for x, y in coords)
    area = f"{left},{top + plot_height} {line} {width - right},{top + plot_height}"
    grid = "".join(f'<line x1="{left}" y1="{top + (1 - value) * plot_height}" x2="{width - right}" '
                   f'y2="{top + (1 - value) * plot_height}" stroke="#dce5eb"/>'
                   f'<text x="{left - 10}" y="{top + (1 - value) * plot_height + 4}" text-anchor="end">{value:g}</text>'
                   for value in (0, .25, .5, .75, 1))
    first = _h(_time(points[0].get("time"), site_timezone))
    last = _h(_time(points[-1].get("time"), site_timezone))
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
            'aria-label="Почасовой прогноз нормализованной мощности от 0 до 1">'
            f'{grid}<polygon points="{area}" fill="#d6f2ea"/><polyline points="{line}" fill="none" '
            'stroke="#087f6e" stroke-width="2.8" stroke-linejoin="round"/>'
            f'<text x="{left}" y="{height - 10}">{first}</text>'
            f'<text x="{width - right}" y="{height - 10}" text-anchor="end">{last}</text></svg>')


def build_forecast_report(result, site_timezone=None):
    """Build a UTF-8 HTML report from the snapshot, safe to open offline."""
    result = _mapping(result)
    requested_zone = site_timezone if site_timezone is not None else result.get("timezone_assumption")
    try:
        site_timezone = ZoneInfo(requested_zone).key
    except (ValueError, KeyError, TypeError):
        site_timezone = "UTC"
    rows = [row for row in (result.get("forecast_sample") or []) if isinstance(row, dict)]
    checks = forecast_checks(result)
    weather, model = _mapping(result.get("weather")), _mapping(result.get("model"))
    quality = _mapping(result.get("data_quality"))
    powers = [row.get("predicted_power") for row in rows]
    valid = bool(powers) and all(_numeric(value) for value in powers)
    avg = sum(powers) / len(powers) if valid else None
    minimum, maximum = (min(powers), max(powers)) if valid else (None, None)
    first, last = (rows[0].get("time"), rows[-1].get("time")) if rows else (None, None)
    head = _pairs([
        ("Турбина", result.get("turbine", "не указана")),
        ("Идентификатор прогноза", result.get("forecast_id", "не указан")),
        ("Период, местное время", f"{_time(first, site_timezone)} — {_time(last, site_timezone)}"),
        ("Период, UTC", f"{_time(first)} — {_time(last)}"),
        ("Момент решения, местное время", _time(result.get("issue_time"), site_timezone)),
        ("Момент решения, UTC", _time(result.get("issue_time"))),
        ("Принятый часовой пояс", result.get("timezone_assumption", "не указан")),
        ("Часовой пояс отображения отчёта", site_timezone if site_timezone == requested_zone else
         f"{site_timezone}; исходный пояс не распознан, отображение в UTC"),
        ("Горизонт", f"{result.get('horizon_hours', 'не указан')} ч; {len(rows)} строк"),
    ])
    cards = "".join(f'<div class="metric"><span>{_h(label)}</span><strong>{_number(value)}</strong></div>'
                    for label, value in (("Средняя мощность", avg), ("Минимальная мощность", minimum),
                                         ("Максимальная мощность", maximum)))
    checks_table = "".join(f'<tr><td>{_h(item["label"])}</td><td><span class="badge {item["status"]}">'
                           f'{STATUS_LABELS[item["status"]]}</span></td><td>{_h(item["detail"])}</td></tr>'
                           for item in checks)
    failed = sum(item["status"] == "fail" for item in checks)
    unknown = sum(item["status"] == "unknown" for item in checks)
    outcome = f"Проверок: {len(checks)}. Ошибок: {failed}. Не подтверждено: {unknown}."
    source = _pairs([
        ("Источник / модель", f"{weather.get('provider', 'не указан')} / {weather.get('model', 'не указана')}"),
        ("Выпуск погодной модели, UTC", _time(weather.get("run_time"))),
        ("Выпуск погодной модели, местное время", _time(weather.get("run_time"), site_timezone)),
        ("Максимальная метка архива, UTC", _time(weather.get("available_at"))),
        ("Правило выбора выпуска", weather.get("selection_policy", "не указано")),
        ("Ячейка погодной сетки", weather.get("grid_cell", "не указана")),
        ("SHA-256 погодных входов", weather.get("input_sha256", "не указан")),
        ("Версия модели мощности", model.get("version", "не указана")),
        ("Первый обучающий час, UTC", _time(model.get("training_start"))),
        ("Последний обучающий час, UTC", _time(model.get("training_end"))),
        ("Доступность последнего обучающего часа, UTC", _time(model.get("training_available_at"))),
        ("Полных часов обучения", model.get("training_hours", "не указано")),
        ("SHA-256 исходного CSV", model.get("source_sha256", "не указан")),
    ])
    quality_labels = {"raw_rows": "Исходных десятиминутных строк", "complete_hours": "Полных часов",
                      "invalid_timestamp_rows": "Некорректных временных меток",
                      "ambiguous_or_nonexistent_clock_rows_removed": "Удалено строк неоднозначного времени",
                      "invalid_value_rows_removed": "Удалено строк с недопустимыми значениями",
                      "empty_hours": "Пустых часов", "incomplete_nonempty_hours_removed": "Удалено неполных часов",
                      "samples_required_per_hour": "Требуемых измерений на час"}
    quality_html = _pairs([(label, quality.get(key, "не указано")) for key, label in quality_labels.items()])
    table_rows = "".join('<tr>' + ''.join(f'<td>{_h(value)}</td>' for value in (
                        _time(row.get("time"), site_timezone), _time(row.get("time")),
                        _number(row.get("predicted_power")), _number(row.get("wind_speed"), 2),
                        _number(row.get("temperature"), 2))) + '</tr>' for row in rows)
    assumptions = list(result.get("assumptions") or [])
    assumptions.extend([
        "Часовой пояс и база нормализации мощности не подтверждены организаторами; исходная шкала — безразмерная [0, 1].",
        "HTTP Last-Modified — свидетельство архива, а не независимый журнал первого опубликования погодного прогноза.",
        "Сеточный прогноз погоды не является измерением на турбине; его пространственная точность ограничена.",
        "Фактическая выработка за февраль отсутствует в предоставленных CSV; этот отчёт не доказывает точность за февраль.",
        "Низкая прогнозная мощность не доказывает метеорологический штиль.",
        "Проверки целостности и времени не являются оценкой качества прогнозирования или гарантией будущей выработки.",
    ])
    limitations = "".join(f"<li>{_h(value)}</li>" for value in dict.fromkeys(str(item) for item in assumptions))
    calibration = model.get("calibration")
    calibration_html = ""
    if isinstance(calibration, dict):
        calibration_html = ('<h3>Калибровка</h3><p>Параметры, сохранённые расчётным модулем:</p><pre>'
                            + _h(json.dumps(calibration, ensure_ascii=False, indent=2, default=str)) + '</pre>')
    pipeline = _pairs([
        ("1 · Данные", f"В снимке указано {model.get('training_hours', 'неизвестно')} полных часов обучения."),
        ("2 · Архив", f"Погодный выпуск {_time(weather.get('run_time'))}; источник {weather.get('provider', 'не указан')}."),
        ("3 · Модель", f"Снимок содержит {len(rows)} почасовых результатов; версия {model.get('version', 'не указана')}."),
        ("4 · Проверка", outcome),
        ("5 · Отчёт", "Сформирован из этого сохранённого снимка; для открытия интернет не требуется."),
    ])
    return f'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>EnergyAI — отчёт {_h(result.get("turbine", ""))}</title>
<style>
:root{{color-scheme:light}}*{{box-sizing:border-box}}body{{margin:0;background:#edf3f5;color:#18303b;font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}}main{{max-width:1120px;margin:32px auto;background:#fff;border-radius:18px;overflow:hidden;box-shadow:0 12px 50px #143b4712}}header{{padding:32px 38px;background:#123c43;color:white}}header p{{color:#bce0dc;margin:8px 0 0}}h1{{font-size:30px;letter-spacing:-.6px;margin:0}}h2{{font-size:21px;margin:0 0 14px}}h3{{font-size:17px}}section{{padding:25px 38px;border-bottom:1px solid #e4edef}}.eyebrow{{font-size:12px;text-transform:uppercase;letter-spacing:2px;color:#8edcc6;margin-bottom:8px}}dl{{margin:0}}dl div{{display:grid;grid-template-columns:minmax(180px,31%) 1fr;gap:16px;padding:7px 0;border-bottom:1px solid #f0f4f6}}dt{{color:#56717d}}dd{{margin:0;overflow-wrap:anywhere}}.metrics{{display:flex;gap:15px;margin:16px 0}}.metric{{flex:1;background:#f0f8f6;border:1px solid #d5eae4;border-radius:12px;padding:16px}}.metric span{{display:block;color:#536f70;font-size:13px}}.metric strong{{font-size:29px;color:#087f6e}}.notice{{background:#fff6df;border-left:4px solid #d69d26;padding:12px 15px;margin:12px 0}}.muted{{color:#5b737c;font-size:13px}}.badge{{display:inline-block;padding:3px 9px;border-radius:20px;font-size:12px;white-space:nowrap}}.pass{{background:#dff4e9;color:#155d42}}.fail{{background:#fee6e5;color:#9d2926}}.unknown{{background:#fff1ce;color:#825817}}table{{width:100%;border-collapse:collapse;font-size:13px}}th{{text-align:left;background:#f0f5f7;font-weight:600}}td,th{{padding:10px 11px;border-bottom:1px solid #e3ebef;vertical-align:top}}.scroll{{overflow:auto}}svg{{width:100%;height:auto}}svg text{{font:11px system-ui,sans-serif;fill:#526d77}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f6f8;padding:16px;border-radius:8px}}li{{margin-bottom:8px}}footer{{padding:20px 38px;color:#67808a;font-size:12px}}@media(max-width:650px){{main{{margin:0;border-radius:0}}header,section{{padding:22px}}dl div{{grid-template-columns:1fr;gap:2px}}.metrics{{flex-direction:column}}h1{{font-size:25px}}}}@media print{{body{{background:white;font-size:11px}}main{{margin:0;box-shadow:none;max-width:none}}header{{background:white;color:#123c43;border-bottom:2px solid #123c43}}header p{{color:#405e64}}section{{padding:18px 22px}}.metrics,svg{{break-inside:avoid}}thead{{display:table-header-group}}tr{{break-inside:avoid}}.scroll{{overflow:visible}}}}
</style></head><body><main>
<header><div class="eyebrow">EnergyAI · исторический прогноз</div><h1>Прогноз мощности · {_h(result.get("turbine", "не указана"))}</h1><p>Почасовой расчёт, происхождение входов и проверка временных ограничений</p></header>
<section><h2>Паспорт прогноза</h2>{head}<p class="notice">Все значения мощности безразмерны. База нормализации не раскрыта; абсолютная мощность и физическая энергия не рассчитываются.</p></section>
<section><h2>Почасовая мощность</h2><div class="metrics">{cards}</div>{_power_svg(rows, site_timezone)}<p class="muted">Ось Y: исходная нормализованная мощность [0, 1]. Время графика: {_h(site_timezone)}. Показатели пересчитаны из почасовых строк.</p></section>
<section><h2>Проверки прогноза</h2><p>{_h(outcome)}</p><div class="scroll"><table><thead><tr><th>Условие</th><th>Результат</th><th>Основание</th></tr></thead><tbody>{checks_table}</tbody></table></div><p class="muted">Проверки вычислены заново по сохранённому снимку. Отсутствующие метаданные не считаются успешной проверкой.</p></section>
<section><h2>Данные → архив → модель → проверка → отчёт</h2><p class="muted">Факты из сохранённого результата; это не журнал времени выполнения этапов.</p>{pipeline}</section>
<section><h2>Происхождение данных и модель</h2>{source}<p class="notice">Метка архива — максимум HTTP Last-Modified использованных объектов. Она не заменяет независимый журнал первого опубликования.</p>{calibration_html}</section>
<section><h2>Качество SCADA</h2><p class="muted">Показатели исходного файла целиком. Для обучения применяется отдельное отсечение по моменту решения.</p>{quality_html}</section>
<section><h2>Почасовая таблица</h2><div class="scroll"><table><thead><tr><th>Местное время</th><th>UTC</th><th>Мощность [0, 1]</th><th>Ветер, м/с</th><th>Температура, °C</th></tr></thead><tbody>{table_rows}</tbody></table></div></section>
<section><h2>Ограничения и допущения</h2><ul>{limitations}</ul></section>
<footer>EnergyAI · автономный HTML-отчёт. Не содержит API-ключей, внешних скриптов или подключений к CDN. Печать и сохранение в PDF доступны через меню браузера.</footer>
</main></body></html>'''
