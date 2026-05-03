"""Промпты извлечения триплетов (русские few-shot, system без markdown)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ...slots.ontology import (
    CANONICAL_TO_RU_LABEL,
    DEFAULT_USER_SLOTS,
    RU_SLOT_LABELS_ORDERED,
    TRIPLET_CLARIFY_EVENTS_TRAVEL_CHAIN,
    TRIPLET_CLARIFY_FAMILY_ROMANCE_BOUNDARY,
    TRIPLET_CLARIFY_KINSHIP_FOREIGN_FAMILY,
    triplet_prompt_show_clarification,
)
from .prompt_fewshots import (
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
        Если передано (даже пустой список) - добавляется контекстный блок.
        None означает "без контекста" (старое поведение).
    enable_deletion : bool
        Расширить схему ответа полем "delete" для явных сигналов удаления.
        Автоматически True когда existing_triplets is not None.
    """
    _ = ontology_slots or DEFAULT_USER_SLOTS.slot_names
    slots_catalog_json = json.dumps(RU_SLOT_LABELS_ORDERED, ensure_ascii=False)

    ru_slot = CANONICAL_TO_RU_LABEL.get(slot_name, slot_name) if slot_name else None

    use_context = existing_triplets is not None
    use_deletion = enable_deletion  # controlled explicitly; independent from use_context

    slot_header = ""
    if slot_name and ru_slot:
        slot_header = (
            f"Текущий слот: {ru_slot} ({slot_name}).\n"
            f"Извлекай только факты, относящиеся к слоту «{ru_slot}».\n"
            f"Факты, принадлежащие другим слотам - не включай, даже если они есть в сообщении.\n"
            f"Если в сообщении нет фактов для слота «{ru_slot}» - верни {{\"triplets\":[]}}.\n"
            "В JSON не указывай поле slot - слот задаётся системой.\n\n"
        )

    # --- Условные блоки уточнений (наборы слотов в ontology.py) ---
    slot_boundary_block = ""
    if triplet_prompt_show_clarification(slot_name, TRIPLET_CLARIFY_FAMILY_ROMANCE_BOUNDARY):
        slot_boundary_block = (
            "ГРАНИЦА СЛОТОВ «СЕМЬЯ» и «РОМАНТИКА»:\n"
            "  СЕМЬЯ (FAMILY) - кровные родственники и законные члены семьи САМОГО ПОЛЬЗОВАТЕЛЯ:\n"
            "    муж, жена, сын, дочь, мама, папа, брат, сестра, бабушка, дедушка,\n"
            "    тёща, тесть, свёкор, свекровь, племянник, внук и т.п.\n"
            "  РОМАНТИКА (ROMANCE) - романтические и любовные отношения пользователя:\n"
            "    парень, девушка, бойфренд, подруга, любовный интерес, бывший/-ая и т.п.\n"
            "  ПРАВИЛО: жена/муж → СЕМЬЯ; девушка/парень → РОМАНТИКА. Никогда не наоборот.\n"
            "  ЧУЖАЯ СЕМЬЯ (семья друга, коллеги, другого человека)\n"
            "    → НЕ ВКЛЮЧАЕТСЯ в слот СЕМЬЯ.\n"
            "    Пример: «у моего друга есть сестра» - НЕ добавлять в СЕМЬЯ пользователя, добавь в друзья, если текущий слот ДРУЗЬЯ.\n\n"
        )

    events_travel_chain_block = ""
    if triplet_prompt_show_clarification(slot_name, TRIPLET_CLARIFY_EVENTS_TRAVEL_CHAIN):
        events_travel_chain_block = (
            "Для событий и поездок - цепочка (место/событие становится субъектом своих атрибутов):\n"
            "  пользователь → действие → место или событие\n"
            "  место или событие → атрибут → значение\n"
            "  Запрещено: {\"subject\":\"пользователь\",\"relation\":\"поездка\",\"object\":\"токио сентябрь\"}\n"
            "  Верно:     {\"subject\":\"пользователь\",\"relation\":\"поездка\",\"object\":\"токио\"}\n"
            "             {\"subject\":\"токио\",\"relation\":\"дата\",\"object\":\"сентябрь\"}\n"
            "             {\"subject\":\"токио\",\"relation\":\"едет с\",\"object\":\"семья\"}\n\n"
        )

    kinship_foreign_family_block = ""
    if triplet_prompt_show_clarification(slot_name, TRIPLET_CLARIFY_KINSHIP_FOREIGN_FAMILY):
        kinship_foreign_family_block = (
            "ТОЧНОСТЬ РОДСТВЕННЫХ СВЯЗЕЙ:\n"
            "  Чётко определяй, КТО кому и КЕМ является. Субъект - тот, у кого есть это родство.\n"
            "  Если речь идёт о родственнике не пользователя, а кого-то другого, не пиши что это родственник пользователя."
            "  Если речь о чужой семье (семье друга, коллеги, другого человека) -"
            "  НЕ добавляй эти факты в слот СЕМЬЯ пользователя, добавляй это в слот ДРУЗЬЯ, если текущий слот ДРУЗЬЯ.\n\n"
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
                    "  1. Если новый факт заменяет старый - добавь его в \"triplets\" И добавь\n"
                    "     старый факт в \"delete\". Для сохранения истории добавь в \"triplets\"\n"
                    "     факт с префиксом «бывшее/прежнее» (пример: «бывшее место жительства»).\n"
                    "  2. Если пользователь явно отменяет факт без замены - добавь старый факт\n"
                    "     в \"delete\". Для сохранения истории добавь в \"triplets\"\n"
                    "     факт с префиксом «бывшее/прежнее» (пример: «бывшее место жительства»).\n"
                    "  3. Если факт просто уточняется - обнови через \"delete\" + новый \"triplets\".\n"
                    "  4. Если новое сообщение не меняет известные факты - \"delete\":[].\n"
                    "  5. Не дублируй уже существующие факты в \"triplets\".\n\n"
                )
            else:
                # context-aware mode without deletion: model sees existing facts
                # but should only output new/changed triplets, no delete field
                context_instructions = (
                    "ИНСТРУКЦИИ ПО РАБОТЕ С ТЕКУЩИМИ ФАКТАМИ:\n"
                    "  1. НЕ ДУБЛИРУЙ уже существующие факты в \"triplets\" - добавляй только новые.\n"
                    "  2. Если факт изменился - добавь новый факт и при необходимости\n"
                    "     исторический («бывшее/прежнее»). Старый будет удалён системой автоматически.\n"
                    "  3. Если сообщение не добавляет новых фактов - верни {\"triplets\":[]}.\n\n"
                )
            context_block = (
                f"Текущие факты в слоте"
                + (f" «{ru_slot}»" if ru_slot else "")
                + " (Не дублируй их, если нет новой информации):\n"
                + facts_lines + "\n\n"
                + context_instructions
            )
        else:
            context_block = (
                "Текущие факты в слоте"
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
            "  inf  - имя, пол, национальность, члены семьи, питомцы, устойчивые привычки (кофе по утрам)\n"
            "  1y   - работа, учёба, жильё, здоровье (диагнозы), авто, местоположение\n"
            "  6m   - хобби, спорт, предпочтения, психическое состояние, знакомства\n"
            "  3m   - цели, романтические отношения, финансовые планы\n"
            "  1m   - расписание, планы на ближайшее будущее, еда\n"
            "  2w   - конкретные события (был на свадьбе, сдал экзамен)\n"
            "  1d   - суточные/общие состояния (стрессовый день, день прошёл спокойно, чувствую себя плохо)\n"
            "  12h  - состояния в рамках дня (в хорошем настроении, подавлена, продуктивна)\n"
            "  6h   - краткосрочные состояния (переутомлена, на подъёме, выгорела за день, ложусь спать)\n"
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
            "\nПОЛЕ \"delete\" - список фактов для явного удаления из памяти.\n"
            "Добавляй в \"delete\" только факты из текущего списка фактов слота.\n"
            "Если нечего удалять - \"delete\":[].\n"
        )

    system = (
        slot_header
        + slot_boundary_block
        + "Ты система извлечения фактов из реплики пользователя.\n"
        "Представь факты как триплеты: субъект, связь, объект.\n"
        "Субъект, связь и объект пиши строчными буквами (lowercase).\n"
        "Не используй символ подчёркивания «_» - разделяй слова только пробелами.\n"
        "Для фактов о самом пользователе используй субъект: пользователь.\n"

        "Если упомянута связанная сущность (питомец, член семьи, коллега и т.п.):\n"
        "  1. Триплет связи - всегда добавляй роль, только потом имя, если оно есть:\n"
        "     Запрещено: {\"subject\":\"пользователь\",\"relation\":\"есть кот\",\"object\":\"рыжик\"}\n"
        "     Верно:     {\"subject\":\"пользователь\",\"relation\":\"есть кот\",\"object\":\"кот пользователя\"}\n"
        "                {\"subject\":\"кот пользователя\",\"relation\":\"имя\",\"object\":\"рыжик\"}\n"
        "  2. Свойства сущности - всегда добавляй роль, только потом имя, если оно есть:\n"
        "     Запрещено: {\"subject\":\"рыжик\",\"relation\":\"болен\",\"object\":\"да\"}\n"
        "     Верно:     {\"subject\":\"кот пользователя\",\"relation\":\"имя\",\"object\":\"рыжик\"}\n"
        "                {\"subject\":\"рыжик\",\"relation\":\"состояние\",\"object\":\"болен\"}\n\n"
        + events_travel_chain_block
        + "Каждый триплет обособлен - понятен при одиночном прочтении, без соседних триплетов.\n"
        "  Субъект и объект должны однозначно называть сущность, даже вне контекста:\n"
        "  Запрещено: {\"subject\":\"старший\",\"relation\":\"имя\",\"object\":\"алёша\"}\n"
        "             (непонятно, чей «старший»)\n"
        "  Верно:     {\"subject\":\"старший сын пользователя\",\"relation\":\"имя\",\"object\":\"алёша\"}\n"
        "Не упаковывай несколько фактов в один объект.\n"
        "Связь однозначно описывает смысл - используй цепочку:\n"
        "  Запрещено: {\"subject\":\"пользователь\",\"relation\":\"частота\",\"object\":\"раз в неделю\"}\n"
        "  Верно:     {\"subject\":\"пользователь\",\"relation\":\"ходит\",\"object\":\"рыбалка\"}\n"
        "             {\"subject\":\"рыбалка\",\"relation\":\"частота\",\"object\":\"раз в неделю\"}\n\n"
        + kinship_foreign_family_block
        + "Не выдумывай факты - только то, что явно сказано в сообщении.\n"
        "Строго запрещено - не создавай записи, которые содержат информацию о пользователе, но не относятся к текущему слоту.\n"
        "Не записывай отсутствие информации - если факт не упомянут или неизвестен,\n"
        "  НЕ создавай триплеты вида «пользователь | хобби | неизвестно»,\n"
        "  Строго запрещено:«пользователь | возраст | не указан» и т.п. Только конкретные факты.\n\n"
        "ОЧЕНЬ ВАЖНО: Факт может быть указан без объекта, если он очевиден из контекста, такие факты нужно добавлять.\n"
        "В сообщении факт может быть указан косвенно, нужно понять его из контекста и добавить.\n"
        "В сообщении может быть не только факт, но и вопрос, рассуждения, эмоции, оценки и т.п. В таком случае обязатенльно нужно добавить факты, даже если он упомянут не напрямую.\n"
        + ttl_block
        + delete_block
        + context_block
        + f"Онтология слотов (справочно): {slots_catalog_json}\n"
        "Ответ только валидный json, без markdown, без пояснений.\n"
        "Схема ответа:\n"
        f"{output_schema}\n"
        f"Максимум триплетов: {max_triplets}.\n"
        'Если нет фактов: {"triplets":[]}.\n\n'
        "ОЧЕНЬ ВАЖНО: Факт может быть указан без объекта, если он очевиден из контекста, такие факты нужно добавлять.\n"
        "В сообщении факт может быть указан косвенно, нужно понять его из контекста и добавить.\n"
        "В сообщении может быть не только факт, но и вопрос, рассуждения, эмоции, оценки и т.п. В таком случае обязатенльно нужно добавить факты, даже если он упомянут не напрямую.\n"
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
        slot_part = f"Слот: {ru_slot}\n" if ru_slot else ""
        action = "Извлеки новые/изменённые факты и укажи факты для удаления." if use_deletion else "Извлеки только новые факты, не дублируй существующие."
        return (
            f"{slot_part}"
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
