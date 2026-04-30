"""Промпты извлечения триплетов (русские few-shot, system без markdown)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..slots.ontology import CANONICAL_TO_RU_LABEL, DEFAULT_USER_SLOTS, RU_SLOT_LABELS_ORDERED
from .prompt_fewshots_ru import (
    triplet_context_few_shot_messages,
    triplet_per_slot_few_shot_messages,
    triplet_single_pass_few_shot_messages,
)


def build_triplet_messages(
    user_message: str,
    *,
    slot_name: str | None = None,
    include_slot: bool = False,
    ontology_slots: List[str] | None = None,
    max_triplets: int = 12,
    ttl_mode: str = "mode2",
    existing_triplets: Optional[List[str]] = None,
    enable_deletion: bool = False,
) -> List[Dict[str, Any]]:
    """
    Построить чат-сообщения для экстракции триплетов.

    Parameters
    ----------
    existing_triplets : list of str or None
        Текущие активные факты слота в формате "subject | relation | object".
        Если передано (даже пустой список) — добавляется контекстный блок.
        None означает "без контекста" (старое поведение).
    enable_deletion : bool
        Расширить схему ответа полем "delete" для явных сигналов удаления.
        Автоматически True когда existing_triplets is not None.
    """
    _ = ontology_slots or DEFAULT_USER_SLOTS.slot_names
    slots_ru_json = json.dumps(RU_SLOT_LABELS_ORDERED, ensure_ascii=False)

    ru_slot = CANONICAL_TO_RU_LABEL.get(slot_name, slot_name) if slot_name else None

    use_context = existing_triplets is not None
    use_deletion = enable_deletion  # controlled explicitly; independent from use_context

    slot_header = ""
    if slot_name and ru_slot:
        slot_header = (
            f"ТЕКУЩИЙ СЛОТ: {ru_slot} ({slot_name}).\n"
            f"ИЗВЛЕКАЙ ИСКЛЮЧИТЕЛЬНО ФАКТЫ, ОТНОСЯЩИЕСЯ К СЛОТУ «{ru_slot}».\n"
            f"ФАКТЫ, ПРИНАДЛЕЖАЩИЕ ДРУГИМ СЛОТАМ — НЕ ВКЛЮЧАЙ, ДАЖЕ ЕСЛИ ОНИ ЕСТЬ В СООБЩЕНИИ.\n"
            f"ЕСЛИ В СООБЩЕНИИ НЕТ ФАКТОВ ДЛЯ СЛОТА «{ru_slot}» — ВЕРНИ {{\"triplets\":[]}}.\n"
            "В JSON НЕ УКАЗЫВАЙ ПОЛЕ slot — СЛОТ ЗАДАЁТСЯ СИСТЕМОЙ.\n\n"
        )

    # --- Блок разграничения FAMILY / ROMANCE ---
    slot_boundary_block = ""
    if slot_name in ("FAMILY", "ROMANCE"):
        slot_boundary_block = (
            "ГРАНИЦА СЛОТОВ «СЕМЬЯ» и «РОМАНТИКА»:\n"
            "  СЕМЬЯ (FAMILY) — кровные родственники и законные члены семьи САМОГО ПОЛЬЗОВАТЕЛЯ:\n"
            "    муж, жена, сын, дочь, мама, папа, брат, сестра, бабушка, дедушка,\n"
            "    тёща, тесть, свёкор, свекровь, племянник, внук и т.п.\n"
            "  РОМАНТИКА (ROMANCE) — романтические и любовные отношения пользователя:\n"
            "    парень, девушка, бойфренд, подруга, любовный интерес, бывший/-ая и т.п.\n"
            "  ПРАВИЛО: жена/муж → СЕМЬЯ; девушка/парень → РОМАНТИКА. Никогда не наоборот.\n"
            "  ЧУЖАЯ СЕМЬЯ (семья друга, коллеги, другого человека)\n"
            "    → НЕ ВКЛЮЧАЕТСЯ в слот СЕМЬЯ пользователя.\n"
            "    Пример: «у моего друга есть сестра» — НЕ добавлять в СЕМЬЯ пользователя.\n\n"
        )

    # --- Блок контекста текущих фактов ---
    context_block = ""
    if use_context:
        if existing_triplets:
            facts_lines = "\n".join(f"  {line}" for line in existing_triplets)
            if use_deletion:
                # llm_inline mode: model should output delete signals
                context_instructions = (
                    "ИНСТРУКЦИИ ПО РАБОТЕ С ТЕКУЩИМИ ФАКТАМИ:\n"
                    "  1. ЕСЛИ НОВЫЙ ФАКТ ЗАМЕНЯЕТ СУЩЕСТВУЮЩИЙ — добавь его в \"triplets\" И добавь\n"
                    "     старый факт в \"delete\". Для сохранения истории добавь в \"triplets\"\n"
                    "     факт с префиксом «бывшее/прежнее» (пример: «бывшее место жительства»).\n"
                    "  2. ЕСЛИ ПОЛЬЗОВАТЕЛЬ ЯВНО ОТМЕНЯЕТ ФАКТ БЕЗ ЗАМЕНЫ — добавь старый факт\n"
                    "     в \"delete\". Для сохранения истории добавь в \"triplets\"\n"
                    "     факт с префиксом «бывшее/прежнее» (пример: «бывшее место жительства»).\n"
                    "  3. ЕСЛИ ФАКТ ПРОСТО УТОЧНЯЕТСЯ — обнови через \"delete\" + новый \"triplets\".\n"
                    "  4. ЕСЛИ НОВОЕ СООБЩЕНИЕ НЕ МЕНЯЕТ ИЗВЕСТНЫЕ ФАКТЫ — \"delete\":[].\n"
                    "  5. НЕ ДУБЛИРУЙ УЖЕ СУЩЕСТВУЮЩИЕ ФАКТЫ В \"triplets\".\n\n"
                )
            else:
                # context-aware mode without deletion: model sees existing facts
                # but should only output new/changed triplets, no delete field
                context_instructions = (
                    "ИНСТРУКЦИИ ПО РАБОТЕ С ТЕКУЩИМИ ФАКТАМИ:\n"
                    "  1. НЕ ДУБЛИРУЙ уже существующие факты в \"triplets\" — добавляй только новые.\n"
                    "  2. Если факт изменился — добавь новый факт и при необходимости\n"
                    "     исторический («бывшее/прежнее»). Старый будет удалён системой автоматически.\n"
                    "  3. Если сообщение не добавляет новых фактов — верни {\"triplets\":[]}.\n\n"
                )
            context_block = (
                f"ТЕКУЩИЕ ФАКТЫ В СЛОТЕ"
                + (f" «{ru_slot}»" if ru_slot else "")
                + " (УЖЕ СОХРАНЕНЫ В ПАМЯТИ):\n"
                + facts_lines + "\n\n"
                + context_instructions
            )
        else:
            context_block = (
                "ТЕКУЩИЕ ФАКТЫ В СЛОТЕ"
                + (f" «{ru_slot}»" if ru_slot else "")
            )

    use_ttl = (ttl_mode == "mode2")

    if use_ttl:
        if include_slot:
            output_schema = (
                '{"triplets":[{"slot":"РАБОТА","subject":"пользователь","relation":"работает как","object":"инженер","ttl":"1y"}]}'
                if not use_deletion else
                '{"triplets":[{"slot":"РАБОТА","subject":"пользователь","relation":"работает как","object":"инженер","ttl":"1y"}],"delete":[{"subject":"пользователь","relation":"работает как","object":"водитель"}]}'
            )
        else:
            output_schema = (
                '{"triplets":[{"subject":"пользователь","relation":"работает как","object":"водитель такси","ttl":"1y"}]}'
                if not use_deletion else
                '{"triplets":[{"subject":"пользователь","relation":"место жительства","object":"сызрань","ttl":"1y"},{"subject":"пользователь","relation":"бывшее место жительства","object":"москва","ttl":"1y"}],"delete":[{"subject":"пользователь","relation":"место жительства","object":"москва"}]}'
            )
        ttl_block = (
            "\nДОПОЛНИТЕЛЬНО К КАЖДОМУ ТРИПЛЕТУ ДОБАВЛЯЙ ПОЛЕ TTL (время жизни факта).\n"
            "ДОПУСТИМЫЕ ЗНАЧЕНИЯ TTL: 6h, 12h, 1d, 3d, 10d, 2w, 3w, 1m, 3m, 6m, 1y, inf\n"
            "ПРАВИЛА ВЫБОРА TTL:\n"
            "  inf  — имя, пол, национальность, члены семьи, питомцы, устойчивые привычки (кофе по утрам)\n"
            "  1y   — работа, учёба, жильё, здоровье (диагнозы), авто, местоположение\n"
            "  6m   — хобби, спорт, предпочтения, психическое состояние, знакомства\n"
            "  3m   — цели, романтические отношения, финансовые планы\n"
            "  1m   — расписание, планы на ближайшее будущее, еда\n"
            "  2w   — конкретные события (был на свадьбе, сдал экзамен)\n"
            "  1d   — суточные/общие состояния (стрессовый день, день прошёл спокойно, чувствую себя плохо)\n"
            "  12h  — состояния в рамках дня (в хорошем настроении, подавлена, продуктивна)\n"
            "  6h   — краткосрочные состояния (переутомлена, на подъёме, выгорела за день)\n"
        )
    else:
        if include_slot:
            output_schema = (
                '{"triplets":[{"slot":"РАБОТА","subject":"пользователь","relation":"работает как","object":"инженер"}]}'
                if not use_deletion else
                '{"triplets":[{"slot":"РАБОТА","subject":"пользователь","relation":"работает как","object":"инженер"}],"delete":[{"subject":"пользователь","relation":"работает как","object":"водитель"}]}'
            )
        else:
            output_schema = (
                '{"triplets":[{"subject":"пользователь","relation":"работает как","object":"водитель такси"}]}'
                if not use_deletion else
                '{"triplets":[{"subject":"пользователь","relation":"место жительства","object":"сызрань"}],"delete":[{"subject":"пользователь","relation":"место жительства","object":"москва"}]}'
            )
        ttl_block = ""

    delete_block = ""
    if use_deletion:
        delete_block = (
            "\nПОЛЕ \"delete\" — список фактов для явного удаления из памяти.\n"
            "Добавляй в \"delete\" только факты из текущего списка фактов слота.\n"
            "Если нечего удалять — \"delete\":[].\n"
        )

    system = (
        slot_header
        + slot_boundary_block
        + context_block
        + "ТЫ СИСТЕМА ИЗВЛЕЧЕНИЯ ФАКТОВ ИЗ РЕПЛИКИ ПОЛЬЗОВАТЕЛЯ.\n"
        "ПРЕДСТАВЬ ФАКТЫ КАК ТРИПЛЕТЫ: СУБЪЕКТ, СВЯЗЬ, ОБЪЕКТ.\n"
        "СУБЪЕКТ, СВЯЗЬ И ОБЪЕКТ ПИШИ СТРОЧНЫМИ БУКВАМИ (lowercase).\n"
        "НЕ ИСПОЛЬЗУЙ СИМВОЛ ПОДЧЁРКИВАНИЯ «_» — РАЗДЕЛЯЙ СЛОВА ТОЛЬКО ПРОБЕЛАМИ.\n"
        "ДЛЯ ФАКТОВ О САМОМ ПОЛЬЗОВАТЕЛЕ ИСПОЛЬЗУЙ СУБЪЕКТ: пользователь.\n"

        "ЕСЛИ УПОМЯНУТА СВЯЗАННАЯ СУЩНОСТЬ (питомец, член семьи, коллега и т.п.):\n"
        "  1. ТРИПЛЕТ СВЯЗИ — объект всегда РОЛЬ, никогда имя:\n"
        "     ЗАПРЕЩЕНО: {\"subject\":\"пользователь\",\"relation\":\"есть кот\",\"object\":\"рыжик\"}\n"
        "     ВЕРНО:     {\"subject\":\"пользователь\",\"relation\":\"есть кот\",\"object\":\"кот пользователя\"}\n"
        "  2. СВОЙСТВА СУЩНОСТИ — субъект всегда РОЛЬ, никогда имя:\n"
        "     ЗАПРЕЩЕНО: {\"subject\":\"рыжик\",\"relation\":\"болен\",\"object\":\"да\"}\n"
        "     ВЕРНО:     {\"subject\":\"кот пользователя\",\"relation\":\"болен\",\"object\":\"да\"}\n"
        "     ВЕРНО:     {\"subject\":\"кот пользователя\",\"relation\":\"имя\",\"object\":\"рыжик\"}\n"

        "ДЛЯ СОБЫТИЙ И ПОЕЗДОК — ЦЕПОЧКА (место/событие становится субъектом своих атрибутов):\n"
        "  пользователь → ДЕЙСТВИЕ → МЕСТО_ИЛИ_СОБЫТИЕ\n"
        "  МЕСТО_ИЛИ_СОБЫТИЕ → атрибут → значение\n"
        "  ЗАПРЕЩЕНО: {\"subject\":\"пользователь\",\"relation\":\"поездка\",\"object\":\"токио сентябрь\"}\n"
        "  ВЕРНО:     {\"subject\":\"пользователь\",\"relation\":\"поездка\",\"object\":\"токио\"}\n"
        "             {\"subject\":\"токио\",\"relation\":\"дата\",\"object\":\"сентябрь\"}\n"
        "             {\"subject\":\"токио\",\"relation\":\"едет с\",\"object\":\"семья\"}\n"

        "КАЖДЫЙ ТРИПЛЕТ ОБОСОБЛЕН — понятен при одиночном прочтении, без соседних триплетов.\n"
        "  Субъект и объект должны однозначно называть сущность, даже вне контекста:\n"
        "  ЗАПРЕЩЕНО: {\"subject\":\"старший\",\"relation\":\"имя\",\"object\":\"алёша\"}\n"
        "             (непонятно, чей «старший»)\n"
        "  ВЕРНО:     {\"subject\":\"старший сын пользователя\",\"relation\":\"имя\",\"object\":\"алёша\"}\n"
        "НЕ УПАКОВЫВАЙ НЕСКОЛЬКО ФАКТОВ В ОДИН OBJECT.\n"
        "RELATION ОДНОЗНАЧНО ОПИСЫВАЕТ СМЫСЛ — используй цепочку:\n"
        "  ЗАПРЕЩЕНО: {\"subject\":\"пользователь\",\"relation\":\"частота\",\"object\":\"раз в неделю\"}\n"
        "  ВЕРНО:     {\"subject\":\"пользователь\",\"relation\":\"ходит\",\"object\":\"рыбалка\"}\n"
        "             {\"subject\":\"рыбалка\",\"relation\":\"частота\",\"object\":\"раз в неделю\"}\n"

        "ТОЧНОСТЬ РОДСТВЕННЫХ СВЯЗЕЙ:\n"
        "  Чётко определяй, КТО кому и КЕМ является. Субъект — тот, у кого есть это родство.\n"
        "  ЗАПРЕЩЕНО: {\"subject\":\"мама пользователя\",\"relation\":\"есть сын\",\"object\":\"пользователь\"}\n"
        "  ВЕРНО:     {\"subject\":\"пользователь\",\"relation\":\"есть мама\",\"object\":\"мама пользователя\"}\n"
        "  Если речь о чужой семье (семье друга, коллеги, другого человека) —\n"
        "  НЕ добавляй эти факты в слот СЕМЬЯ пользователя.\n"

        "НЕ ВЫДУМЫВАЙ ФАКТЫ — только то, что явно сказано в сообщении.\n"
        "НЕ ЗАПИСЫВАЙ ОТСУТСТВИЕ ИНФОРМАЦИИ — если факт не упомянут или неизвестен,\n"
        "  НЕ создавай триплеты вида «пользователь | хобби | неизвестно»,\n"
        "  «пользователь | возраст | не указан» и т.п. Только конкретные факты.\n"
        "ИГНОРИРУЙ ЧИСТЫЕ ЭМОЦИИ БЕЗ ПРОВЕРЯЕМЫХ ФАКТОВ.\n"
        + ttl_block
        + delete_block
        + f"ОНТОЛОГИЯ СЛОТОВ (СПРАВОЧНО): {slots_ru_json}\n"
        "ОТВЕТ ТОЛЬКО ВАЛИДНЫЙ JSON. БЕЗ MARKDOWN. БЕЗ ТЕКСТА ВНЕ JSON.\n"
        "СХЕМА ОТВЕТА:\n"
        f"{output_schema}\n"
        f"МАКСИМУМ ТРИПЛЕТОВ: {max_triplets}.\n"
        'ЕСЛИ НЕТ УСТОЙЧИВЫХ ФАКТОВ: {"triplets":[]}.'
    )

    def user_turn_no_slot(msg: str) -> str:
        return f"Сообщение пользователя:\n{msg}\n\nИзвлеки триплеты."

    def user_turn_with_slot(msg: str) -> str:
        return (
            f"Слот: {ru_slot}\n"
            f"Сообщение пользователя:\n{msg}\n\n"
            f"Извлеки триплеты только для слота «{ru_slot}»."
        )

    def user_turn_with_context(msg: str) -> str:
        facts_str = "\n".join(existing_triplets) if existing_triplets else "(нет фактов)"
        slot_part = f"Слот: {ru_slot}\n" if ru_slot else ""
        action = "Извлеки новые/изменённые факты и укажи факты для удаления." if use_deletion else "Извлеки только новые факты, не дублируй существующие."
        return (
            f"{slot_part}"
            f"Текущие факты:\n{facts_str}\n\n"
            f"Сообщение пользователя:\n{msg}\n\n"
            f"{action}"
        )

    if use_context:
        few_shot = triplet_context_few_shot_messages(
            user_turn_with_context, slot_name=slot_name, use_ttl=use_ttl,
            enable_deletion=use_deletion,
        )
        user_turn = user_turn_with_context
    elif include_slot:
        few_shot = triplet_single_pass_few_shot_messages(user_turn_no_slot, use_ttl=use_ttl)
        user_turn = user_turn_no_slot
    elif slot_name and ru_slot:
        few_shot = triplet_per_slot_few_shot_messages(
            user_turn_no_slot, user_turn_with_slot, slot_name, use_ttl=use_ttl
        )
        user_turn = user_turn_with_slot
    else:
        few_shot = triplet_per_slot_few_shot_messages(
            user_turn_no_slot, user_turn_no_slot, None, use_ttl=use_ttl
        )
        user_turn = user_turn_no_slot

    return (
        [{"role": "system", "content": system}]
        + few_shot
        + [{"role": "user", "content": user_turn(user_message)}]
    )
