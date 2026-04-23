"""
support_playbooks.py
Núcleo conductual de NeuroGuIA.

Objetivo:
- Proveer rutas claras, humanas y consistentes para acompañamiento.
- Evitar improvisación caótica en situaciones frecuentes.
- Permitir que otra capa (LLM o renderer) reformule con calidez,
  pero SIN cambiar la intención, la seguridad ni la ruta base.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Literal


# =========================================================
# Tipos base
# =========================================================

Domain = Literal[
    "crisis",
    "ansiedad",
    "bloqueo_ejecutivo",
    "sueno",
    "sobrecarga_cuidador",
    "pregunta_simple",
    "meta_question",
    "validacion_emocional",
    "rechazo_estrategia",
    "depresion_baja_energia",
    "meditacion_guiada",
    "clarificacion",
    "cierre",
    "general",
]

TurnFamily = Literal[
    "new_request",
    "followup_acceptance",
    "clarification_request",
    "blocked_followup",
    "specific_action_request",
    "literal_phrase_request",
    "post_action_followup",
    "simple_question",
    "validation_request",
    "strategy_rejection",
    "outcome_report",
    "meta_question",
    "closure_or_pause",
]

OutcomePolarity = Literal[
    "no_change",
    "worse",
    "partial_relief",
    "improved",
    "unknown",
]


@dataclass
class UserSignal:
    """Señales ya detectadas por capas anteriores o inferidas localmente."""
    domain: Domain
    turn_family: TurnFamily
    outcome: OutcomePolarity = "unknown"
    user_text: str = ""
    asks_for_meds: bool = False
    asks_for_phrase: bool = False
    asks_for_next_step: bool = False
    expresses_confusion: bool = False
    expresses_overwhelm: bool = False
    expresses_rejection: bool = False
    expresses_impossibility: bool = False
    wants_to_pause: bool = False
    wants_to_continue: bool = False
    mentions_risk: bool = False


@dataclass
class ResponsePlan:
    """
    Plan conductual base.
    Luego el LLM o renderer puede reformularlo, pero no cambiar su intención.
    """
    goal: str
    tone: str
    validation: str
    main_response: str
    optional_followup: Optional[str] = None
    next_step: Optional[str] = None
    literal_phrase: Optional[str] = None
    micro_practice: Optional[str] = None
    safety_note: Optional[str] = None
    close_softly: bool = False
    needs_professional_redirect: bool = False
    tags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PlaybookSpec:
    route_id: Domain
    tone_objective: str
    validation_base: str
    max_steps: int
    expected_user_responses: List[str]
    if_not_understood: TurnFamily
    if_rejected: TurnFamily
    if_continue: TurnFamily
    if_pause: TurnFamily


# =========================================================
# Utilidades
# =========================================================

def normalize_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def contains_any(text: str, phrases: Sequence[str]) -> bool:
    t = normalize_text(text)
    return any(p in t for p in phrases)


# =========================================================
# Interceptores críticos de seguridad / límites
# =========================================================

MED_REQUEST_MARKERS = [
    "pastilla", "pastillas", "medicina", "medicamento", "medicamentos",
    "dosis", "que tomo", "qué tomo", "que me tomo", "qué me tomo",
    "que pastilla", "qué pastilla", "que me recomiendas tomar",
    "qué me recomiendas tomar", "recetame", "recétame",
    "algo para dormir", "algo para calmarme", "algo para la ansiedad",
    "que le doy", "qué le doy", "que le puedo dar", "qué le puedo dar",
]

HIGH_RISK_MARKERS = [
    "quiero morirme", "me quiero morir", "hacerme daño", "hacerme dano",
    "lastimarme", "lastimarlo", "lastimarla", "suicid", "matarme",
    "matarlo", "matarla", "no puedo mantenerlo seguro", "no puedo mantenerla segura",
]


def intercept_medication_request(signal: UserSignal) -> Optional[ResponsePlan]:
    if signal.asks_for_meds or contains_any(signal.user_text, MED_REQUEST_MARKERS):
        return ResponsePlan(
            goal="safe_medication_boundary",
            tone="calido_claro_seguro",
            validation="Entiendo que estás buscando algo que ayude ya.",
            main_response=(
                "No puedo decirte qué pastilla tomar ni recomendar medicamentos o dosis. "
                "Si esto te está afectando mucho, lo más seguro es consultarlo con un profesional de salud."
            ),
            optional_followup=(
                "Si quieres, sí puedo ayudarte con una medida no farmacológica sencilla para esta noche o este momento."
            ),
            needs_professional_redirect=True,
            tags=["safety", "medication_boundary"],
        )
    return None


def intercept_high_risk(signal: UserSignal) -> Optional[ResponsePlan]:
    if signal.mentions_risk or contains_any(signal.user_text, HIGH_RISK_MARKERS):
        return ResponsePlan(
            goal="high_risk_redirect",
            tone="calido_firme_seguro",
            validation="Gracias por decirlo. Esto no conviene cargarlo sola/o.",
            main_response=(
                "Ahora lo importante es la seguridad inmediata. Busca apoyo presencial o de emergencia "
                "cerca de ti en este momento. Si hay alguien contigo o a quien puedas llamar, hazlo ya."
            ),
            optional_followup=(
                "Mientras llega apoyo, aléjate de objetos peligrosos y trata de no quedarte sola/o si puedes."
            ),
            safety_note="prioridad_seguridad_inmediata",
            needs_professional_redirect=True,
            tags=["safety", "high_risk"],
        )
    return None


# =========================================================
# Playbooks específicos
# =========================================================

def playbook_meta_question(signal: UserSignal) -> ResponsePlan:
    text = normalize_text(signal.user_text)

    if "como puedo llamarte" in text or "cómo puedo llamarte" in text:
        return ResponsePlan(
            goal="answer_about_system_briefly",
            tone="calido_humano_directo",
            validation="",
            main_response="Puedes decirme NeuroGuIA, o como te salga más natural.",
            close_softly=True,
            tags=["meta", "identity"],
        )

    if contains_any(text, ["puedo platicar contigo", "puedo hablar contigo"]):
        return ResponsePlan(
            goal="answer_about_system_briefly",
            tone="calido_humano_directo",
            validation="",
            main_response="Sí, claro. Puedes platicar conmigo cuando lo necesites y lo vemos paso a paso.",
            close_softly=True,
            tags=["meta", "availability"],
        )

    return ResponsePlan(
        goal="answer_about_system_briefly",
        tone="calido_humano_directo",
        validation="",
        main_response=(
            "Soy NeuroGuIA, un sistema de acompañamiento conversacional pensado para ayudar "
            "a madres, cuidadores y personas neurodivergentes con orientación clara, apoyo emocional y pasos concretos."
        ),
        optional_followup="Si quieres, puedo acompañarte con una duda puntual o con algo que te esté pesando ahora.",
        tags=["meta", "identity"],
    )


def playbook_crisis(signal: UserSignal) -> ResponsePlan:
    if signal.turn_family == "closure_or_pause" or signal.wants_to_pause:
        return ResponsePlan(
            goal="pause_after_crisis_step",
            tone="calido_suave",
            validation="Esta bien.",
            main_response="Podemos parar aqui por ahora. Si vuelve a subir, regresa a una sola frase breve y a bajar estimulos alrededor.",
            close_softly=True,
            tags=["crisis", "pause"],
        )

    if signal.turn_family == "clarification_request" or signal.expresses_confusion:
        return ResponsePlan(
            goal="clarify_crisis_step",
            tone="claro_firme",
            validation="Si, te lo digo mas simple.",
            main_response="Haz solo una cosa: baja una fuente de ruido, gente o exigencia.",
            optional_followup="Despues manten una frase breve y no discutas.",
            tags=["crisis", "clarify"],
        )

    if signal.turn_family == "literal_phrase_request" or signal.asks_for_phrase:
        return ResponsePlan(
            goal="literal_phrase_for_crisis",
            tone="calido_firme_breve",
            validation="",
            main_response="Puedes decirle esto, tal cual:",
            literal_phrase="Estoy aquí contigo. No hace falta hablar mucho ahora. Vamos a bajar esto juntos.",
            optional_followup="Después de decirlo, mantén la frase breve y no agregues más instrucciones al mismo tiempo.",
            tags=["crisis", "literal_phrase"],
        )

    if signal.turn_family == "specific_action_request":
        return ResponsePlan(
            goal="first_concrete_step_in_crisis",
            tone="firme_claro",
            validation="Estoy contigo.",
            main_response=(
                "Empieza por una sola cosa: baja una fuente de ruido, exigencia o gente alrededor."
            ),
            optional_followup=(
                "Si eso ya está hecho, mantén distancia segura y evita discutir o razonar en pleno pico."
            ),
            tags=["crisis", "first_step"],
        )

    if signal.turn_family == "followup_acceptance" or signal.asks_for_next_step or signal.wants_to_continue:
        return ResponsePlan(
            goal="next_step_after_initial_crisis_containment",
            tone="firme_claro",
            validation="Bien, seguimos sin meter demasiadas cosas.",
            main_response="El siguiente paso es sostener una sola frase breve y mirar si baja un poco la tension.",
            optional_followup="Si no baja nada, cambiamos una sola cosa mas del entorno.",
            tags=["crisis", "next_step"],
        )

    if signal.turn_family == "post_action_followup":
        return ResponsePlan(
            goal="check_effect_or_next_step_crisis",
            tone="firme_claro",
            validation="",
            main_response=(
                "Ahora mira solo esto: ¿hay menos tensión, menos ruido o más espacio seguro que hace un momento?"
            ),
            optional_followup=(
                "Si bajó un poco, sostén eso y no metas otra indicación todavía. "
                "Si no bajó nada, cambia solo una cosa más del entorno."
            ),
            tags=["crisis", "followup"],
        )

    if signal.turn_family == "outcome_report":
        if signal.outcome == "improved":
            return ResponsePlan(
                goal="hold_after_effect_crisis",
                tone="calido_firme",
                validation="Bien, aunque sea un poco, eso ya importa.",
                main_response=(
                    "Sostén lo que ayudó y no agregues otra demanda por ahora."
                ),
                close_softly=True,
                tags=["crisis", "hold"],
            )
        if signal.outcome in ("no_change", "worse"):
            return ResponsePlan(
                goal="change_modality_after_no_effect_crisis",
                tone="firme_claro",
                validation="Gracias por decirlo.",
                main_response=(
                    "Entonces cambia de vía: menos palabras y más espacio seguro. "
                    "Mueve la situación a un punto con menos gente o menos ruido, si se puede."
                ),
                optional_followup="Si necesitas, también puedo darte una frase breve para ese momento.",
                tags=["crisis", "switch_mode"],
            )

    return ResponsePlan(
        goal="initial_crisis_containment",
        tone="calido_firme_breve",
        validation="Estoy contigo.",
        main_response=(
            "Lo primero es bajar demanda alrededor. Empieza por una sola fuente de estímulo cercana."
        ),
        optional_followup="Si quieres, después vemos el siguiente paso sin hacerlo todo a la vez.",
        tags=["crisis", "initial"],
    )


def playbook_anxiety(signal: UserSignal) -> ResponsePlan:
    if signal.turn_family == "closure_or_pause" or signal.wants_to_pause:
        return ResponsePlan(
            goal="pause_after_regulation_attempt",
            tone="calido_suave",
            validation="Esta bien.",
            main_response="Podemos dejarlo aqui por ahora. Si despues vuelve a apretar, retoma solo una exhalacion larga o una nota breve.",
            close_softly=True,
            tags=["ansiedad", "pause"],
        )

    if signal.turn_family == "clarification_request" or signal.expresses_confusion:
        return ResponsePlan(
            goal="clarify_anxiety_step",
            tone="claro_calido",
            validation="Si, te lo digo mas simple.",
            main_response="Haz solo esto: apoya los pies en el piso y suelta el aire un poco mas largo una vez.",
            tags=["ansiedad", "clarify"],
        )

    if signal.turn_family == "blocked_followup" or signal.expresses_overwhelm:
        return ResponsePlan(
            goal="reduce_anxiety_now",
            tone="calido_contenedor",
            validation="Sí, esto se puede sentir demasiado.",
            main_response=(
                "No tienes que resolver todo ahora. Haz solo esto: apoya los pies en el piso y suelta el aire más largo una vez."
            ),
            optional_followup="Si quieres, luego vemos si esto necesita acción hoy o si puede esperar.",
            micro_practice="grounding_exhale",
            tags=["ansiedad", "grounding"],
        )

    if signal.turn_family == "strategy_rejection":
        return ResponsePlan(
            goal="change_modality_after_no_effect",
            tone="calido_claro",
            validation="Está bien, entonces no seguimos por esa vía.",
            main_response=(
                "Cambiemos de enfoque. En vez de pensar más, dime si te ayuda más una cosa del cuerpo, del entorno o una frase breve."
            ),
            optional_followup="Si no quieres elegir, yo elijo una por ti.",
            tags=["ansiedad", "strategy_switch"],
        )

    if signal.turn_family == "outcome_report":
        if signal.outcome == "partial_relief":
            return ResponsePlan(
                goal="hold_or_close_after_partial_effect",
                tone="calido_claro",
                validation="Bien, aunque sea un poco ya cuenta.",
                main_response=(
                    "Sostén eso y no agregues otra cosa por ahora."
                ),
                close_softly=True,
                tags=["ansiedad", "hold"],
            )
        if signal.outcome == "improved":
            return ResponsePlan(
                goal="close_after_effect",
                tone="calido",
                validation="Qué bueno que aflojó.",
                main_response="Por ahora basta. Puedes quedarte solo con eso y seguir más tarde si hace falta.",
                close_softly=True,
                tags=["ansiedad", "close"],
            )
        if signal.outcome in ("no_change", "worse"):
            return ResponsePlan(
                goal="change_modality_after_no_effect",
                tone="calido_claro",
                validation="Gracias por decirlo.",
                main_response=(
                    "Entonces no seguimos con lo mismo. Haz esto: mira a tu alrededor y nombra tres cosas que ves, sin explicarlas."
                ),
                optional_followup="Después vemos si eso bajó un poco la carga o si cambiamos otra vez.",
                tags=["ansiedad", "switch_modality"],
            )

    if signal.turn_family == "followup_acceptance" or signal.asks_for_next_step or signal.wants_to_continue:
        return ResponsePlan(
            goal="choose_one_pressure_after_grounding",
            tone="calido_directo",
            validation="Bien, ya no vamos con todo junto.",
            main_response="Ahora elige una sola presion real para hoy. Si quieres, escribe una frase con eso y deja lo demas quieto.",
            optional_followup="Si no quieres elegir, te ayudo a decidir por vencimiento o por carga.",
            tags=["ansiedad", "next_step"],
        )

    if signal.turn_family == "specific_action_request":
        return ResponsePlan(
            goal="one_real_next_step_for_anxiety",
            tone="calido_directo",
            validation="",
            main_response=(
                "Haz una sola cosa visible: abre una nota y escribe una frase con lo que más te aprieta ahora."
            ),
            optional_followup="No hace falta resolverlo todavía, solo sacarlo un poco de la cabeza.",
            tags=["ansiedad", "visible_step"],
        )

    return ResponsePlan(
        goal="initial_anxiety_support",
        tone="calido_contendor",
        validation="Tiene sentido que esto te esté pesando.",
        main_response=(
            "Vamos despacio. Primero baja un poco la activación: apoya los pies en el piso y suelta el aire más largo una vez."
        ),
        optional_followup="Si luego quieres, vemos qué parte sí necesita acción hoy.",
        tags=["ansiedad", "initial"],
    )


def playbook_executive_block(signal: UserSignal) -> ResponsePlan:
    if signal.turn_family == "closure_or_pause" or signal.wants_to_pause:
        return ResponsePlan(
            goal="pause_after_block_step",
            tone="calido_suave",
            validation="Esta bien.",
            main_response="Aqui podemos parar por ahora. Cuando vuelvas, retoma solo desde dejar el material abierto.",
            close_softly=True,
            tags=["bloqueo", "pause"],
        )

    if signal.turn_family == "clarification_request" or signal.expresses_confusion:
        return ResponsePlan(
            goal="simplify_blocking_step",
            tone="claro_calido",
            validation="Sí, te lo digo más simple.",
            main_response="Solo haz esto: abre el archivo, cuaderno o material. Nada más.",
            tags=["bloqueo", "clarify"],
        )

    if signal.expresses_impossibility or signal.turn_family == "blocked_followup":
        return ResponsePlan(
            goal="lower_demand_for_block",
            tone="calido_claro",
            validation="No tienes que poder con todo ahora.",
            main_response=(
                "Haz lo más pequeño posible: mueve la mano al material o deja el archivo abierto. Solo eso."
            ),
            optional_followup="Si ni eso sale, aquí podemos parar un momento y bajar más la exigencia.",
            tags=["bloqueo", "lower_demand"],
        )

    if signal.turn_family == "specific_action_request":
        return ResponsePlan(
            goal="first_visible_step",
            tone="claro_directo",
            validation="",
            main_response=(
                "Empieza aquí: abre solo el material que toca y deja el cursor o la hoja lista."
            ),
            optional_followup="No pienses todavía en terminar; solo en dejar una salida visible.",
            tags=["bloqueo", "first_step"],
        )

    if signal.turn_family == "strategy_rejection":
        return ResponsePlan(
            goal="replace_rejected_strategy",
            tone="calido_claro",
            validation="Está bien, entonces no seguimos por ahí.",
            main_response=(
                "Cambiemos de vía: dime solo el nombre de la materia o tarea, y yo te ayudo a partirla."
            ),
            optional_followup="Si prefieres, también puedo darte yo el primer paso sin que tengas que decidir.",
            tags=["bloqueo", "strategy_switch"],
        )

    if signal.turn_family == "followup_acceptance" or signal.asks_for_next_step or signal.wants_to_continue:
        return ResponsePlan(
            goal="next_step_after_opening_material",
            tone="claro_directo",
            validation="Bien, seguimos pequeno.",
            main_response="El siguiente paso es dejar una salida visible: escribe solo el titulo o la primera linea, aunque quede simple.",
            optional_followup="No busques hacerlo bien todavia. Solo visible.",
            tags=["bloqueo", "next_step"],
        )

    return ResponsePlan(
        goal="initial_block_support",
        tone="calido_practico",
        validation="Sí, esto puede bloquear mucho.",
        main_response="Empieza aquí: abre solo el archivo, cuaderno o material que toca.",
        optional_followup="Con eso alcanza por ahora.",
        tags=["bloqueo", "initial"],
    )


def playbook_sleep(signal: UserSignal) -> ResponsePlan:
    if signal.turn_family == "closure_or_pause" or signal.wants_to_pause:
        return ResponsePlan(
            goal="pause_sleep_support",
            tone="calido_suave",
            validation="Esta bien.",
            main_response="Podemos dejarlo aqui por ahora. Si luego quieres retomarlo, volvemos desde una sola bajada de estimulo.",
            close_softly=True,
            tags=["sueno", "pause"],
        )

    if signal.turn_family == "clarification_request":
        return ResponsePlan(
            goal="clarify_sleep_problem",
            tone="claro_calido",
            validation="Sí, vamos a aterrizarlo mejor.",
            main_response=(
                "Dime solo qué pesa más: te cuesta apagar la mente, el cuerpo sigue activado, o el entorno no ayuda a dormir."
            ),
            tags=["sueno", "clarify"],
        )

    if signal.turn_family == "specific_action_request":
        return ResponsePlan(
            goal="one_sleep_step",
            tone="calido_directo",
            validation="",
            main_response=(
                "Empieza por una sola cosa antes de dormir: baja una fuente de estímulo, como luz, ruido o pantalla."
            ),
            optional_followup="Después podemos ver si hace falta otra medida sencilla.",
            tags=["sueno", "first_step"],
        )

    if signal.turn_family == "outcome_report":
        if signal.outcome in ("no_change", "worse"):
            return ResponsePlan(
                goal="change_sleep_modality",
                tone="calido_claro",
                validation="Gracias por decirlo.",
                main_response=(
                    "Entonces cambiemos de vía. En vez de seguir intentando dormir ya, haz una bajada gradual de 5 a 10 minutos sin pantalla ni exigencia."
                ),
                optional_followup="Y si el tema te está afectando mucho, lo más seguro es comentarlo con un profesional de salud.",
                tags=["sueno", "switch"],
            )
        if signal.outcome in ("partial_relief", "improved"):
            return ResponsePlan(
                goal="hold_sleep_gain",
                tone="calido",
                validation="Bien, eso ya da una pista.",
                main_response="Sostén lo que ayudó y no agregues más cosas por ahora.",
                close_softly=True,
                tags=["sueno", "hold"],
            )

    if signal.turn_family == "followup_acceptance" or signal.asks_for_next_step or signal.wants_to_continue:
        return ResponsePlan(
            goal="next_sleep_step_after_reducing_stimulus",
            tone="calido_suave",
            validation="Bien, vamos con una sola capa mas.",
            main_response="Despues de bajar estimulos, deja una bajada gradual de 5 a 10 minutos sin pantalla ni exigencia.",
            optional_followup="Si quieres, luego vemos si el problema pesa mas en la mente, el cuerpo o el entorno.",
            tags=["sueno", "next_step"],
        )

    return ResponsePlan(
        goal="initial_sleep_support",
        tone="calido_suave",
        validation="Sí, el sueño puede mover todo lo demás.",
        main_response=(
            "Vamos con algo concreto: antes de dormir, baja una sola fuente de estímulo como luz, ruido o pantalla."
        ),
        optional_followup="Si quieres, luego vemos si el problema es mente acelerada, cuerpo activado o entorno.",
        tags=["sueno", "initial"],
    )


def playbook_caregiver_overload(signal: UserSignal) -> ResponsePlan:
    if signal.turn_family == "strategy_rejection":
        return ResponsePlan(
            goal="replace_relief_path_for_caregiver",
            tone="calido_claro",
            validation="Esta bien, no seguimos por ahi.",
            main_response="Cambiemos de via. En vez de elegir entre todo, dime si pesa mas el ruido, las decisiones o el cansancio.",
            optional_followup="Si no quieres elegir, yo te ayudo a dejar una sola carga quieta por ahora.",
            tags=["cuidador", "strategy_switch"],
        )

    if signal.turn_family == "followup_acceptance" or signal.asks_for_next_step or signal.wants_to_continue:
        return ResponsePlan(
            goal="release_one_load_for_caregiver",
            tone="calido_practico",
            validation="Bien, solo una cosa mas.",
            main_response="El siguiente paso es soltar una carga concreta por ahora: una decision, una tarea o una exigencia que pueda esperar.",
            optional_followup="No es abandonar todo; es protegerte un poco para sostener mejor lo importante.",
            tags=["cuidador", "next_step"],
        )

    if signal.turn_family == "closure_or_pause" or signal.wants_to_pause:
        return ResponsePlan(
            goal="pause_caregiver_relief",
            tone="calido_suave",
            validation="Esta bien.",
            main_response="Aqui podemos parar por ahora. Si luego vuelves, retomamos desde una sola carga a bajar.",
            close_softly=True,
            tags=["cuidador", "pause"],
        )

    return ResponsePlan(
        goal="reduce_caregiver_overload",
        tone="calido_contenedor",
        validation="Sí, esto puede sentirse demasiado para una sola persona.",
        main_response=(
            "No tienes que sostener todo al mismo tiempo. Elige una sola carga que podamos bajar ahora: ruido, decisiones o exigencias."
        ),
        optional_followup="Si quieres, te ayudo a elegir una y dejar las demás quietas por un momento.",
        tags=["cuidador", "overload"],
    )


def playbook_validation(signal: UserSignal) -> ResponsePlan:
    return ResponsePlan(
        goal="brief_validation",
        tone="calido_claro",
        validation="",
        main_response="Sí, tiene sentido que esto te esté pesando así.",
        optional_followup="Si quieres, lo vemos con más calma o vamos a algo concreto.",
        close_softly=True,
        tags=["validation"],
    )


def playbook_clarification(signal: UserSignal) -> ResponsePlan:
    return ResponsePlan(
        goal="clarify_in_one_step",
        tone="claro_calido",
        validation="Si, te lo digo mas simple.",
        main_response="Vamos con una sola idea: una cosa a la vez, no todo junto.",
        tags=["clarification"],
    )


def playbook_strategy_rejection(signal: UserSignal) -> ResponsePlan:
    return ResponsePlan(
        goal="change_strategy_without_pressure",
        tone="calido_claro",
        validation="Esta bien, no hace falta seguir por algo que no te esta sirviendo.",
        main_response="Cambiemos de enfoque. Te doy una alternativa mas pequena o una frase breve, sin repetir lo mismo.",
        optional_followup="Si quieres, elijo yo la opcion mas simple y seguimos desde ahi.",
        tags=["strategy_rejection"],
    )


def playbook_next_step(signal: UserSignal) -> ResponsePlan:
    return ResponsePlan(
        goal="offer_next_step_without_reset",
        tone="calido_directo",
        validation="Bien, seguimos paso a paso.",
        main_response="El siguiente paso es hacer una sola cosa pequena y visible, sin abrir otro frente.",
        optional_followup="Si quieres, te lo bajo todavia mas.",
        tags=["next_step"],
    )


def playbook_meditation(signal: UserSignal) -> ResponsePlan:
    return ResponsePlan(
        goal="teach_short_meditation",
        tone="calido_guiado",
        validation="",
        main_response=(
            "Vamos con una práctica breve de un minuto. "
            "Siéntate o quédate como estés. "
            "Apoya los pies si puedes. "
            "Toma aire normal. "
            "Ahora exhala un poco más largo una vez. "
            "Luego mira un punto fijo y cuenta tres exhalaciones sin forzarte."
        ),
        optional_followup="Si quieres, después te guío otra más corta o una para antes de dormir.",
        micro_practice="1_min_breath_anchor",
        tags=["meditacion"],
    )


def playbook_simple_question(signal: UserSignal) -> ResponsePlan:
    return ResponsePlan(
        goal="answer_simple_question_briefly",
        tone="claro_directo",
        validation="",
        main_response="Sí, puedo ayudarte con orientación breve, acompañamiento emocional y pasos concretos según lo que esté pasando.",
        close_softly=True,
        tags=["simple_question"],
    )


def playbook_close(signal: UserSignal) -> ResponsePlan:
    return ResponsePlan(
        goal="closure_or_pause",
        tone="calido_suave",
        validation="Está bien.",
        main_response="Aquí podemos parar por ahora. Si luego necesitas seguir, aquí sigo contigo.",
        close_softly=True,
        tags=["closure"],
    )


PLAYBOOK_SPECS: Dict[Domain, PlaybookSpec] = {
    "crisis": PlaybookSpec(
        route_id="crisis",
        tone_objective="calido_firme_breve",
        validation_base="Estoy contigo.",
        max_steps=3,
        expected_user_responses=["si", "que le digo", "sigue igual", "ya bajo un poco", "paramos aqui"],
        if_not_understood="clarification_request",
        if_rejected="strategy_rejection",
        if_continue="followup_acceptance",
        if_pause="closure_or_pause",
    ),
    "ansiedad": PlaybookSpec(
        route_id="ansiedad",
        tone_objective="calido_contenedor",
        validation_base="Tiene sentido que esto te este pesando.",
        max_steps=3,
        expected_user_responses=["no puedo con todo", "y luego", "no me sirve", "ya aflojo un poco", "paramos aqui"],
        if_not_understood="clarification_request",
        if_rejected="strategy_rejection",
        if_continue="followup_acceptance",
        if_pause="closure_or_pause",
    ),
    "bloqueo_ejecutivo": PlaybookSpec(
        route_id="bloqueo_ejecutivo",
        tone_objective="claro_calido",
        validation_base="Si, esto puede bloquear mucho.",
        max_steps=3,
        expected_user_responses=["no puedo", "que hago", "y luego", "eso no me sirve", "paramos aqui"],
        if_not_understood="clarification_request",
        if_rejected="strategy_rejection",
        if_continue="followup_acceptance",
        if_pause="closure_or_pause",
    ),
    "sueno": PlaybookSpec(
        route_id="sueno",
        tone_objective="calido_suave",
        validation_base="Si, el sueno puede mover todo lo demas.",
        max_steps=3,
        expected_user_responses=["no duermo", "y luego", "sigue igual", "ya bajo un poco", "paramos aqui"],
        if_not_understood="clarification_request",
        if_rejected="strategy_rejection",
        if_continue="followup_acceptance",
        if_pause="closure_or_pause",
    ),
    "sobrecarga_cuidador": PlaybookSpec(
        route_id="sobrecarga_cuidador",
        tone_objective="calido_contenedor",
        validation_base="Si, esto puede sentirse demasiado para una sola persona.",
        max_steps=2,
        expected_user_responses=["ya no puedo con esto", "y luego", "eso no me sirve", "paramos aqui"],
        if_not_understood="clarification_request",
        if_rejected="strategy_rejection",
        if_continue="followup_acceptance",
        if_pause="closure_or_pause",
    ),
    "pregunta_simple": PlaybookSpec(
        route_id="pregunta_simple",
        tone_objective="claro_directo",
        validation_base="",
        max_steps=1,
        expected_user_responses=["que haces", "me ayudas con esto"],
        if_not_understood="clarification_request",
        if_rejected="strategy_rejection",
        if_continue="followup_acceptance",
        if_pause="closure_or_pause",
    ),
    "meta_question": PlaybookSpec(
        route_id="meta_question",
        tone_objective="calido_humano_directo",
        validation_base="",
        max_steps=1,
        expected_user_responses=["quien eres", "como puedo llamarte"],
        if_not_understood="clarification_request",
        if_rejected="strategy_rejection",
        if_continue="followup_acceptance",
        if_pause="closure_or_pause",
    ),
    "clarificacion": PlaybookSpec(
        route_id="clarificacion",
        tone_objective="claro_calido",
        validation_base="Si, te lo digo mas simple.",
        max_steps=1,
        expected_user_responses=["no entiendo", "no se", "como"],
        if_not_understood="clarification_request",
        if_rejected="strategy_rejection",
        if_continue="followup_acceptance",
        if_pause="closure_or_pause",
    ),
    "rechazo_estrategia": PlaybookSpec(
        route_id="rechazo_estrategia",
        tone_objective="calido_claro",
        validation_base="Esta bien, no hace falta seguir por algo que no te esta sirviendo.",
        max_steps=1,
        expected_user_responses=["no me sirve", "otra cosa"],
        if_not_understood="clarification_request",
        if_rejected="strategy_rejection",
        if_continue="followup_acceptance",
        if_pause="closure_or_pause",
    ),
    "cierre": PlaybookSpec(
        route_id="cierre",
        tone_objective="calido_suave",
        validation_base="Esta bien.",
        max_steps=1,
        expected_user_responses=["ya estuvo", "aqui paro"],
        if_not_understood="clarification_request",
        if_rejected="strategy_rejection",
        if_continue="followup_acceptance",
        if_pause="closure_or_pause",
    ),
    "general": PlaybookSpec(
        route_id="general",
        tone_objective="calido_claro",
        validation_base="Aqui estoy contigo.",
        max_steps=2,
        expected_user_responses=["no se", "que sigue", "paramos aqui"],
        if_not_understood="clarification_request",
        if_rejected="strategy_rejection",
        if_continue="followup_acceptance",
        if_pause="closure_or_pause",
    ),
}

PLAYBOOK_BUILDERS: Dict[Domain, Callable[[UserSignal], ResponsePlan]] = {
    "crisis": playbook_crisis,
    "ansiedad": playbook_anxiety,
    "bloqueo_ejecutivo": playbook_executive_block,
    "sueno": playbook_sleep,
    "sobrecarga_cuidador": playbook_caregiver_overload,
    "pregunta_simple": playbook_simple_question,
    "meta_question": playbook_meta_question,
    "clarificacion": playbook_clarification,
    "rechazo_estrategia": playbook_strategy_rejection,
    "cierre": playbook_close,
    "general": playbook_next_step,
}


def get_playbook_spec(route_id: Domain) -> Optional[PlaybookSpec]:
    return PLAYBOOK_SPECS.get(route_id)


def get_playbook_builder(route_id: Domain) -> Optional[Callable[[UserSignal], ResponsePlan]]:
    return PLAYBOOK_BUILDERS.get(route_id)


# =========================================================
# Router principal
# =========================================================

def build_response_plan(signal: UserSignal) -> ResponsePlan:
    """
    Punto principal de entrada.
    Primero intercepta seguridad/límites, luego resuelve playbook.
    """
    high_risk = intercept_high_risk(signal)
    if high_risk:
        return high_risk

    meds = intercept_medication_request(signal)
    if meds:
        return meds

    if signal.turn_family == "closure_or_pause" or signal.domain == "cierre":
        return playbook_close(signal)

    if signal.turn_family == "meta_question" or signal.domain == "meta_question":
        return playbook_meta_question(signal)

    if signal.domain == "crisis":
        return playbook_crisis(signal)

    if signal.domain == "ansiedad":
        return playbook_anxiety(signal)

    if signal.domain == "bloqueo_ejecutivo":
        return playbook_executive_block(signal)

    if signal.domain == "sueno":
        return playbook_sleep(signal)

    if signal.domain == "sobrecarga_cuidador":
        return playbook_caregiver_overload(signal)

    if signal.domain == "meditacion_guiada":
        return playbook_meditation(signal)

    if signal.turn_family == "validation_request":
        return playbook_validation(signal)

    if signal.turn_family == "strategy_rejection" or signal.domain == "rechazo_estrategia":
        return playbook_strategy_rejection(signal)

    if signal.turn_family == "followup_acceptance" or signal.asks_for_next_step:
        return playbook_next_step(signal)

    if signal.turn_family == "simple_question" or signal.domain == "pregunta_simple":
        return playbook_simple_question(signal)

    if signal.domain == "depresion_baja_energia":
        return ResponsePlan(
            goal="support_low_energy_without_forcing",
            tone="calido_suave",
            validation="Sí, esto puede dejarte sin energía hasta para lo pequeño.",
            main_response=(
                "No hace falta empujarte demasiado ahora. Haz solo esto: cambia de postura o apoya los pies en el piso y quédate ahí un momento."
            ),
            optional_followup=(
                "Si quieres, después vemos si hay una sola cosa pequeña que sí sea posible hoy."
            ),
            tags=["baja_energia"],
        )

    if signal.turn_family == "clarification_request" or signal.domain == "clarificacion":
        return playbook_clarification(signal)

    return ResponsePlan(
        goal="general_support",
        tone="calido_claro",
        validation="Aquí estoy contigo.",
        main_response="Cuéntame qué parte pesa más y lo vemos paso a paso.",
        tags=["general"],
    )


# =========================================================
# Helper básico para pruebas rápidas
# =========================================================

def infer_basic_signal(user_text: str, domain: Domain, turn_family: TurnFamily) -> UserSignal:
    """
    Helper básico por si necesitas prototipar rápido.
    No sustituye routers más finos, pero sirve para conectar playbooks.
    """
    text = normalize_text(user_text)

    outcome: OutcomePolarity = "unknown"
    if contains_any(text, ["sigo igual", "no cambio", "no cambió", "no ayudo", "no ayudó"]):
        outcome = "no_change"
    elif contains_any(text, ["empeoro", "empeoró", "peor", "subio", "subió más"]):
        outcome = "worse"
    elif contains_any(text, ["bajo un poco", "me ayudo un poco", "me ayudó un poco", "aflojo un poco"]):
        outcome = "partial_relief"
    elif contains_any(text, ["ya estoy mejor", "ya bajo", "ya bajó", "me ayudo", "me ayudó"]):
        outcome = "improved"

    return UserSignal(
        domain=domain,
        turn_family=turn_family,
        outcome=outcome,
        user_text=user_text,
        asks_for_meds=contains_any(text, MED_REQUEST_MARKERS),
        asks_for_phrase=contains_any(text, ["que frase", "qué frase", "que digo", "qué digo", "que le digo", "qué le digo", "que puedo decirle", "qué puedo decirle"]),
        asks_for_next_step=contains_any(text, ["y luego", "que sigue", "qué sigue", "que mas", "qué más", "y despues", "y después", "y ahora que", "y ahora qué"]),
        expresses_confusion=contains_any(text, ["no entiendo", "no te entiendo", "como", "cómo", "que?", "qué?"]),
        expresses_overwhelm=contains_any(text, ["me gana", "no puedo con todo", "todo se me junta", "me rebasa", "me rebaso"]),
        expresses_rejection=contains_any(text, ["no me sirve", "no me ayuda", "otra cosa", "eso no funciona", "eso no aplica"]),
        expresses_impossibility=contains_any(text, ["no puedo", "no me sale", "no me da", "no logro"]),
        wants_to_pause=contains_any(text, ["por ahora ya", "ya estuvo", "aqui paro", "aquí paro"]),
        wants_to_continue=contains_any(text, ["si", "sí", "ok", "dale", "continua", "continúa"]),
        mentions_risk=contains_any(text, HIGH_RISK_MARKERS),
    )


if __name__ == "__main__":
    examples = [
        infer_basic_signal("Está ocurriendo una crisis y necesito ayuda", "crisis", "new_request"),
        infer_basic_signal("qué le digo?", "crisis", "literal_phrase_request"),
        infer_basic_signal("mejor dime qué pastillas tomar", "sueno", "specific_action_request"),
        infer_basic_signal("quién eres?", "meta_question", "meta_question"),
        infer_basic_signal("no puedo ni empezar", "bloqueo_ejecutivo", "blocked_followup"),
    ]

    for s in examples:
        plan = build_response_plan(s)
        print("=" * 60)
        print("USER:", s.user_text)
        print("GOAL:", plan.goal)
        print("MAIN:", plan.main_response)
        if plan.literal_phrase:
            print("PHRASE:", plan.literal_phrase)
        if plan.optional_followup:
            print("FOLLOWUP:", plan.optional_followup)
