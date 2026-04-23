from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from core.support_playbooks import (
    Domain,
    OutcomePolarity,
    ResponsePlan,
    TurnFamily,
    UserSignal,
    build_response_plan,
    get_playbook_spec,
    infer_basic_signal,
)


GuidanceMode = Literal["advance", "hold", "close", "switch", "direct"]


@dataclass
class SupportFlowResult:
    handled: bool
    route_id: Domain
    conversation_domain: str
    turn_family: TurnFamily
    guidance_mode: GuidanceMode
    continuity_score: float
    continuity_reason: Optional[str]
    outcome: OutcomePolarity
    response_plan: Optional[ResponsePlan] = None
    signal: Optional[UserSignal] = None
    support_flow_state: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "handled": self.handled,
            "route_id": self.route_id,
            "conversation_domain": self.conversation_domain,
            "turn_family": self.turn_family,
            "guidance_mode": self.guidance_mode,
            "continuity_score": self.continuity_score,
            "continuity_reason": self.continuity_reason,
            "outcome": self.outcome,
            "response_plan": self.response_plan,
            "signal": self.signal,
            "support_flow_state": dict(self.support_flow_state),
            "notes": list(self.notes),
        }


class SupportFlowEngine:
    """
    Minimal behavioral flow layer.

    Role:
    - decide the active support route
    - preserve continuity across follow-ups
    - translate the turn into a deterministic playbook call
    - optionally build a synthetic payload compatible with the orchestrator
    """

    ROUTE_TO_CONVERSATION_DOMAIN: Dict[Domain, str] = {
        "crisis": "crisis_activa",
        "ansiedad": "ansiedad_cognitiva",
        "bloqueo_ejecutivo": "disfuncion_ejecutiva",
        "sueno": "sueno_regulacion",
        "sobrecarga_cuidador": "sobrecarga_cuidador",
        "pregunta_simple": "apoyo_general",
        "meta_question": "apoyo_general",
        "validacion_emocional": "apoyo_general",
        "rechazo_estrategia": "apoyo_general",
        "depresion_baja_energia": "apoyo_general",
        "meditacion_guiada": "apoyo_general",
        "clarificacion": "apoyo_general",
        "cierre": "apoyo_general",
        "general": "apoyo_general",
    }

    ROUTE_TO_PHASE: Dict[Domain, str] = {
        "crisis": "containment",
        "ansiedad": "cognitive_unloading",
        "bloqueo_ejecutivo": "micro_start",
        "sueno": "wind_down",
        "sobrecarga_cuidador": "relief",
        "pregunta_simple": "clarification",
        "meta_question": "clarification",
        "validacion_emocional": "clarification",
        "rechazo_estrategia": "clarification",
        "depresion_baja_energia": "clarification",
        "meditacion_guiada": "clarification",
        "clarificacion": "clarification",
        "cierre": "clarification",
        "general": "clarification",
    }

    ROUTE_TO_GOAL: Dict[Domain, str] = {
        "crisis": "contain_and_protect",
        "ansiedad": "reduce_mental_overload",
        "bloqueo_ejecutivo": "enable_first_step",
        "sueno": "stabilize_sleep_transition",
        "sobrecarga_cuidador": "reduce_caregiver_burden",
        "pregunta_simple": "answer_directly",
        "meta_question": "answer_directly",
        "validacion_emocional": "validate_and_stay_present",
        "rechazo_estrategia": "change_without_reset",
        "depresion_baja_energia": "lower_demand",
        "meditacion_guiada": "guide_short_regulation",
        "clarificacion": "clarify_without_reset",
        "cierre": "close_softly",
        "general": "clarify_and_support",
    }

    CATEGORY_TO_ROUTE: Dict[str, Domain] = {
        "crisis_activa": "crisis",
        "escalada_emocional": "crisis",
        "prevencion_escalada": "crisis",
        "regulacion_post_evento": "crisis",
        "ansiedad_cognitiva": "ansiedad",
        "disfuncion_ejecutiva": "bloqueo_ejecutivo",
        "sueno_regulacion": "sueno",
        "sobrecarga_cuidador": "sobrecarga_cuidador",
        "apoyo_general": "general",
    }

    STATE_TO_ROUTE: Dict[str, Domain] = {
        "meltdown": "crisis",
        "shutdown": "crisis",
        "emotional_dysregulation": "crisis",
        "cognitive_anxiety": "ansiedad",
        "general_distress": "ansiedad",
        "executive_dysfunction": "bloqueo_ejecutivo",
        "sleep_disruption": "sueno",
        "burnout": "sobrecarga_cuidador",
        "parental_fatigue": "sobrecarga_cuidador",
    }

    FOLLOWUP_FAMILIES = {
        "followup_acceptance",
        "clarification_request",
        "blocked_followup",
        "specific_action_request",
        "literal_phrase_request",
        "post_action_followup",
        "strategy_rejection",
        "outcome_report",
        "closure_or_pause",
    }

    DIRECT_FAMILIES = {"meta_question", "simple_question"}

    META_MARKERS = [
        "quien eres",
        "quien sos",
        "cual es tu nombre",
        "tu nombre",
        "como puedo llamarte",
        "puedo hablar contigo",
        "puedo platicar contigo",
        "que puedes hacer",
    ]
    CLOSURE_MARKERS = [
        "ya estuvo",
        "aqui paro",
        "aqui paramos",
        "por ahora ya",
        "quiero parar",
        "vamos a parar",
        "ya no quiero seguir",
    ]
    REJECTION_MARKERS = [
        "no me sirve",
        "no me ayuda",
        "eso no funciona",
        "otra cosa",
        "no quiero seguir por ahi",
    ]
    BLOCKED_MARKERS = [
        "no se",
        "no lo se",
        "no se como",
        "no tengo idea",
        "no tengo una idea clara",
        "no puedo",
        "no me sale",
        "no logro",
        "me gana",
        "todo se me junta",
    ]
    CLARIFICATION_MARKERS = [
        "no entiendo",
        "no te entiendo",
        "explicamelo",
        "explicame",
        "mas simple",
        "como",
        "cual",
    ]
    ACTION_CLARIFICATION_MARKERS = [
        "que frase",
        "que digo",
        "que le digo",
        "cual frase",
        "cual paso",
        "linea de que",
        "que linea",
        "que tipo",
        "por donde",
        "como",
        "cual",
    ]
    DIRECT_ACTION_QUESTION_MARKERS = [
        "que",
        "como",
        "cual",
        "por donde",
        "que hago",
        "que sigue",
        "que tipo",
        "que frase",
        "linea de que",
    ]
    POST_ACTION_MARKERS = [
        "ya",
        "ya esta",
        "ya estuvo",
        "ya lo hice",
        "listo",
        "hecho",
        "y despues",
        "y luego",
        "que sigue",
        "que mas",
        "y ahora que",
        "ahora que",
    ]
    NEXT_STEP_MARKERS = [
        "y luego",
        "que sigue",
        "que mas",
        "y despues",
        "y ahora que",
        "seguimos",
        "continua",
        "continuemos",
        "dale",
    ]
    SPECIFIC_ACTION_MARKERS = [
        "que hago",
        "por donde empiezo",
        "como empiezo",
        "dime el paso",
        "dime que hago",
    ]
    SIMPLE_QUESTION_MARKERS = [
        "me ayudas",
        "puedes ayudarme",
        "que haces",
        "sirves para",
    ]
    OUTCOME_IMPROVED_MARKERS = [
        "ya estoy mejor",
        "ya bajo",
        "ya bajo un poco",
        "me ayudo",
        "me ayudo un poco",
        "aflojo",
        "aflojo un poco",
    ]
    OUTCOME_WORSE_MARKERS = [
        "peor",
        "empeoro",
        "empeoro mas",
        "subio",
    ]
    OUTCOME_NO_CHANGE_MARKERS = [
        "sigo igual",
        "no cambio",
        "no ayudo",
        "no funciono",
        "sigue igual",
    ]
    ROUTE_TEXT_MARKERS: Dict[Domain, List[str]] = {
        "crisis": [
            "crisis",
            "gritando",
            "golpeando",
            "no la puedo calmar",
            "no lo puedo calmar",
            "hay riesgo",
        ],
        "ansiedad": [
            "ansiedad",
            "me gana todo",
            "me gana",
            "todo se me junta",
            "muy saturada",
            "muy saturado",
            "muy ansiosa",
            "muy ansioso",
        ],
        "bloqueo_ejecutivo": [
            "no puedo empezar",
            "no puedo arrancar",
            "bloqueada",
            "bloqueado",
            "tarea",
            "archivo",
            "materia",
            "pendiente",
        ],
        "sueno": [
            "no duermo",
            "no puedo dormir",
            "dormir",
            "sueno",
            "desvelo",
            "insomnio",
            "pantalla antes de dormir",
        ],
        "sobrecarga_cuidador": [
            "ya no puedo con esto",
            "cuidar",
            "me pesa cuidar",
            "agotada de cuidar",
            "agotado de cuidar",
        ],
    }
    ACTION_FOLLOWUP_FAMILIES = {
        "followup_acceptance",
        "post_action_followup",
        "blocked_followup",
        "clarification_request",
        "literal_phrase_request",
        "outcome_report",
    }
    COUNTED_ACTION_FOLLOWUP_FAMILIES = {
        "followup_acceptance",
        "post_action_followup",
        "blocked_followup",
        "outcome_report",
    }
    FOLLOWUP_EXIT_GOALS = {"close_temporarily", "decide_one_path", "switch_strategy"}
    FOLLOWUP_EXIT_THRESHOLD = 2

    def resolve_turn(
        self,
        source_message: str,
        effective_message: Optional[str] = None,
        previous_frame: Optional[Dict[str, Any]] = None,
        conversation_frame: Optional[Dict[str, Any]] = None,
        conversation_control: Optional[Dict[str, Any]] = None,
        state_analysis: Optional[Dict[str, Any]] = None,
        category_analysis: Optional[Dict[str, Any]] = None,
        intent_analysis: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> SupportFlowResult:
        del conversation_frame, conversation_control, intent_analysis, chat_history

        previous_frame = previous_frame or {}
        state_analysis = state_analysis or {}
        category_analysis = category_analysis or {}
        text = str(effective_message or source_message or "").strip()

        if not text:
            return SupportFlowResult(
                handled=False,
                route_id="general",
                conversation_domain="apoyo_general",
                turn_family="new_request",
                guidance_mode="direct",
                continuity_score=0.0,
                continuity_reason=None,
                outcome="unknown",
                notes=["empty_message"],
            )

        previous_route = self._resolve_previous_route(previous_frame)
        action_memory = self._extract_action_memory(previous_frame=previous_frame, fallback_route=previous_route)
        turn_family = self._detect_turn_family(text=text, previous_frame=previous_frame)
        route_id = self._resolve_route_id(
            text=text,
            previous_frame=previous_frame,
            previous_route=previous_route,
            turn_family=turn_family,
            state_analysis=state_analysis,
            category_analysis=category_analysis,
        )
        continuity_score, continuity_reason = self._detect_continuity(
            previous_route=previous_route,
            route_id=route_id,
            turn_family=turn_family,
            previous_frame=previous_frame,
        )
        handled = self._is_covered(
            route_id=route_id,
            previous_route=previous_route,
            turn_family=turn_family,
            continuity_score=continuity_score,
        )

        outcome = self._detect_outcome(text)
        signal = infer_basic_signal(
            user_text=text,
            domain=route_id,
            turn_family=turn_family,
        )
        signal.outcome = outcome
        signal.asks_for_phrase = turn_family == "literal_phrase_request"
        signal.asks_for_next_step = turn_family in {"followup_acceptance", "post_action_followup"}
        signal.expresses_confusion = turn_family in {"clarification_request", "literal_phrase_request"}
        signal.expresses_rejection = turn_family == "strategy_rejection"
        signal.expresses_impossibility = turn_family == "blocked_followup"
        signal.wants_to_pause = turn_family == "closure_or_pause"
        signal.wants_to_continue = turn_family in {"followup_acceptance", "post_action_followup"}
        response_plan = (
            self._build_contextual_response_plan(
                signal=signal,
                route_id=route_id,
                previous_frame=previous_frame,
                action_memory=action_memory,
            )
            if handled
            else None
        )
        guidance_mode = self._resolve_guidance_mode(
            turn_family=turn_family,
            outcome=outcome,
            response_plan=response_plan,
        )

        spec = get_playbook_spec(route_id)
        previous_state = dict(previous_frame.get("support_flow_state") or {})
        step_index = self._next_step_index(
            previous_state=previous_state,
            route_id=route_id,
            guidance_mode=guidance_mode,
            continuity_score=continuity_score,
            max_steps=spec.max_steps if spec else 1,
        )
        conversation_domain = self.ROUTE_TO_CONVERSATION_DOMAIN.get(route_id, "apoyo_general")
        action_state = self._resolve_action_state(
            response_plan=response_plan,
            route_id=route_id,
            conversation_domain=conversation_domain,
            previous_action=action_memory,
        )
        followup_trace = self._resolve_followup_trace(
            previous_state=previous_state,
            turn_family=turn_family,
            route_id=route_id,
            continuity_score=continuity_score,
            action_memory=action_memory,
            response_plan=response_plan,
            outcome=outcome,
            guidance_mode=guidance_mode,
        )
        support_flow_state = {
            "active": handled,
            "handled_by": "support_flow_engine",
            "route_id": route_id,
            "conversation_domain": conversation_domain,
            "turn_family": turn_family,
            "guidance_mode": guidance_mode,
            "continuity_score": continuity_score,
            "continuity_reason": continuity_reason,
            "step_index": step_index,
            "max_steps": spec.max_steps if spec else 1,
            "goal": response_plan.goal if response_plan else None,
            "close_softly": bool(response_plan.close_softly) if response_plan else False,
            "playbook_tags": list(response_plan.tags) if response_plan else [],
            "last_action_instruction": action_state.get("last_action_instruction"),
            "last_action_type": action_state.get("last_action_type"),
            "last_action_goal": action_state.get("last_action_goal"),
            "last_action_domain": action_state.get("last_action_domain"),
            "action_followup_count": followup_trace.get("action_followup_count", 0),
            "recent_followup_modes": list(followup_trace.get("recent_followup_modes", [])),
            "followup_exit": followup_trace.get("followup_exit"),
        }

        notes: List[str] = []
        if previous_route:
            notes.append(f"previous_route:{previous_route}")
        if continuity_reason:
            notes.append(f"continuity:{continuity_reason}")
        if followup_trace.get("followup_exit"):
            notes.append(f"followup_exit:{followup_trace['followup_exit']}")

        return SupportFlowResult(
            handled=handled,
            route_id=route_id,
            conversation_domain=conversation_domain,
            turn_family=turn_family,
            guidance_mode=guidance_mode,
            continuity_score=continuity_score,
            continuity_reason=continuity_reason,
            outcome=outcome,
            response_plan=response_plan,
            signal=signal,
            support_flow_state=support_flow_state,
            notes=notes,
        )

    def build_orchestrator_payloads(self, result: SupportFlowResult) -> Dict[str, Any]:
        if not result.handled or not result.response_plan:
            return {}

        response_text = self.render_response_text(result.response_plan)
        selected_microaction = self._selected_microaction(result.response_plan)
        response_shape = self._response_shape(result)
        stage_name = "guided_support_flow"
        phase = self.ROUTE_TO_PHASE.get(result.route_id, "clarification")
        intervention_level = self._intervention_level(result)
        should_close = result.guidance_mode == "close" or bool(result.response_plan.close_softly)

        response_goal = {
            "goal": result.response_plan.goal,
            "strategy_signature": f"support_flow:{result.route_id}:{result.response_plan.goal}",
            "response_shape": response_shape,
            "form_variant": result.guidance_mode,
            "intervention_level": intervention_level,
            "candidate_actions": self._candidate_actions(result.response_plan),
            "literal_phrase_candidates": [result.response_plan.literal_phrase] if result.response_plan.literal_phrase else [],
            "possible_questions": [],
            "should_offer_question": False,
            "followup_policy": "avoid",
            "selected_microaction": selected_microaction,
            "selected_strategy": result.response_plan.goal,
            "selected_routine_type": None,
            "suggested_content": [result.response_plan.main_response],
            "priority_order": [result.route_id, result.guidance_mode],
            "intervention_type": "guided_support_flow",
            "keep_minimal": should_close,
        }

        decision_payload = {
            "decision_mode": "support_flow_engine",
            "intervention_type": "guided_support_flow",
            "selected_strategy": result.response_plan.goal,
            "selected_microaction": selected_microaction,
            "selected_routine_type": None,
            "priority_order": response_goal["priority_order"],
            "avoid": [],
            "decision_flags": {
                "handled_by_support_flow_engine": True,
                "guidance_mode": result.guidance_mode,
                "turn_family": result.turn_family,
            },
            "response_goal": response_goal,
            "response_plan": response_goal,
            "reuse_response_candidate": None,
        }

        stage_result = {
            "stage": stage_name,
            "conversation_domain": result.conversation_domain,
            "conversation_phase": phase,
            "continuity_phase": phase,
            "phase_changed": result.continuity_score >= 0.8,
            "phase_progression_reason": result.guidance_mode,
            "turn_type": result.turn_family,
            "turn_family": result.turn_family,
            "clarification_mode": (
                "support_flow" if result.turn_family in {"clarification_request", "literal_phrase_request"} else None
            ),
            "crisis_guided_mode": result.route_id == "crisis",
            "domain_shift": {"detected": False, "target_domain": result.conversation_domain},
            "intervention_level": intervention_level,
            "stuck_followup_count": int(result.support_flow_state.get("action_followup_count", 0) or 0),
            "progression_signals": {
                "guidance_mode": result.guidance_mode,
                "continuity_score": result.continuity_score,
                "action_followup_count": int(result.support_flow_state.get("action_followup_count", 0) or 0),
                "recent_followup_modes": list(result.support_flow_state.get("recent_followup_modes", [])),
            },
            "should_close_with_followup": not should_close,
        }

        confidence_payload = {
            "overall_confidence": 0.93 if result.continuity_score >= 0.8 else 0.88,
            "confidence_level": "high",
            "source": "support_flow_engine",
            "reason": "deterministic_playbook_flow",
        }

        fallback_payload = {
            "use_llm": False,
            "fallback_reason": "handled_by_support_flow_engine",
            "prompt_mode": "guided_support_flow",
            "should_learn_if_good": False,
        }

        llm_policy = {
            "should_use_llm": False,
            "reason": "handled_by_support_flow_engine",
            "prompt_mode": "guided_support_flow",
            "domain": result.conversation_domain,
            "phase": phase,
            "category": result.conversation_domain,
            "intent": "guided_support",
        }

        conversational_intent = {
            "rhythm": "steady",
            "pressure": "low",
            "permissiveness": "high" if result.turn_family in {"blocked_followup", "closure_or_pause"} else "moderate",
            "closing_style": "soft_close" if should_close else "none",
        }

        response_package = {
            "response": response_text,
            "text": response_text,
            "mode": "guided_support_flow",
            "suggested_strategy": decision_payload["selected_strategy"],
            "suggested_microaction": selected_microaction,
            "suggested_question": None,
            "response_metadata": {
                "source": "support_flow_engine",
                "route_id": result.route_id,
                "turn_family": result.turn_family,
                "guidance_mode": result.guidance_mode,
                "support_flow_state": dict(result.support_flow_state),
            },
        }

        stage_hints = {
            "source": "support_flow_engine",
            "route_id": result.route_id,
            "turn_family": result.turn_family,
            "guidance_mode": result.guidance_mode,
        }

        conversation_control_updates = {
            "turn_type": result.turn_family,
            "turn_family": result.turn_family,
            "phase": phase,
            "domain": result.conversation_domain,
            "phase_progression_reason": result.guidance_mode,
            "last_action_instruction": result.support_flow_state.get("last_action_instruction"),
            "last_action_type": result.support_flow_state.get("last_action_type"),
            "last_action_goal": result.support_flow_state.get("last_action_goal"),
            "last_action_domain": result.support_flow_state.get("last_action_domain"),
            "support_flow_state": dict(result.support_flow_state),
        }

        conversation_frame_updates = {
            "conversation_domain": result.conversation_domain,
            "support_goal": self.ROUTE_TO_GOAL.get(result.route_id, "clarify_and_support"),
            "conversation_phase": phase,
            "turn_type": result.turn_family,
            "turn_family": result.turn_family,
            "continuity_score": result.continuity_score,
            "phase_progression_reason": result.guidance_mode,
            "last_action_instruction": result.support_flow_state.get("last_action_instruction"),
            "last_action_type": result.support_flow_state.get("last_action_type"),
            "last_action_goal": result.support_flow_state.get("last_action_goal"),
            "last_action_domain": result.support_flow_state.get("last_action_domain"),
            "support_flow_state": dict(result.support_flow_state),
        }

        return {
            "stage_result": stage_result,
            "stage_hints": stage_hints,
            "confidence_payload": confidence_payload,
            "decision_payload": decision_payload,
            "fallback_payload": fallback_payload,
            "llm_policy": llm_policy,
            "conversational_intent": conversational_intent,
            "response_package": response_package,
            "conversation_control_updates": conversation_control_updates,
            "conversation_frame_updates": conversation_frame_updates,
        }

    def render_response_text(self, response_plan: ResponsePlan) -> str:
        parts: List[str] = []
        if response_plan.validation:
            parts.append(response_plan.validation.strip())
        if response_plan.main_response:
            parts.append(response_plan.main_response.strip())
        if response_plan.literal_phrase:
            parts.append(f'"{response_plan.literal_phrase.strip()}"')
        if response_plan.optional_followup:
            parts.append(response_plan.optional_followup.strip())
        return " ".join(part for part in parts if part).strip()

    def _resolve_previous_route(self, previous_frame: Dict[str, Any]) -> Optional[Domain]:
        support_state = previous_frame.get("support_flow_state") or {}
        route_id = support_state.get("route_id")
        if route_id:
            return route_id

        previous_domain = str(previous_frame.get("conversation_domain") or "").strip()
        return self.CATEGORY_TO_ROUTE.get(previous_domain)

    def _resolve_route_id(
        self,
        text: str,
        previous_frame: Dict[str, Any],
        previous_route: Optional[Domain],
        turn_family: TurnFamily,
        state_analysis: Dict[str, Any],
        category_analysis: Dict[str, Any],
    ) -> Domain:
        if turn_family == "meta_question":
            return "meta_question"
        if turn_family == "simple_question" and not previous_route:
            return "pregunta_simple"
        if turn_family == "closure_or_pause" and not previous_route:
            return "cierre"
        if turn_family == "clarification_request" and not previous_route:
            return "clarificacion"
        if turn_family == "strategy_rejection" and not previous_route:
            return "rechazo_estrategia"

        if previous_route and turn_family in self.FOLLOWUP_FAMILIES:
            return previous_route

        route_from_category = self.CATEGORY_TO_ROUTE.get(str(category_analysis.get("detected_category") or "").strip())
        if route_from_category and route_from_category != "general":
            return route_from_category

        route_from_state = self.STATE_TO_ROUTE.get(str(state_analysis.get("primary_state") or "").strip())
        if route_from_state and route_from_state != "general":
            return route_from_state

        normalized = self._normalize(text)
        for route_id, markers in self.ROUTE_TEXT_MARKERS.items():
            if self._contains_any(normalized, markers):
                return route_id

        if turn_family == "blocked_followup":
            previous_domain = str(previous_frame.get("conversation_domain") or "").strip()
            return self.CATEGORY_TO_ROUTE.get(previous_domain, previous_route or "general")

        return previous_route or "general"

    def _detect_turn_family(self, text: str, previous_frame: Dict[str, Any]) -> TurnFamily:
        normalized = self._normalize(text)
        action_memory = self._extract_action_memory(previous_frame=previous_frame, fallback_route=self._resolve_previous_route(previous_frame))
        has_active_action = self._has_active_action(action_memory)

        if self._contains_any(normalized, self.META_MARKERS):
            return "meta_question"
        if self._contains_any(normalized, self.CLOSURE_MARKERS):
            return "closure_or_pause"
        if self._contains_any(normalized, self.REJECTION_MARKERS):
            return "strategy_rejection"
        if self._looks_like_outcome_report(normalized):
            return "outcome_report"
        if has_active_action and self._looks_like_current_action_clarification(normalized):
            if self._wants_literal_phrase(normalized=normalized, action_memory=action_memory):
                return "literal_phrase_request"
            return "clarification_request"
        if has_active_action and self._is_post_action_followup(normalized):
            return "post_action_followup"
        if self._contains_any(normalized, self.CLARIFICATION_MARKERS):
            return "clarification_request"
        if self._contains_any(normalized, self.BLOCKED_MARKERS):
            return "blocked_followup"
        if self._contains_any(normalized, self.NEXT_STEP_MARKERS):
            if has_active_action:
                return "post_action_followup"
            return "followup_acceptance"
        if self._contains_any(normalized, self.SPECIFIC_ACTION_MARKERS):
            return "specific_action_request"
        if has_active_action and normalized.endswith("?") and self._looks_like_direct_action_question(normalized):
            if self._wants_literal_phrase(normalized=normalized, action_memory=action_memory):
                return "literal_phrase_request"
            return "clarification_request"
        if self._contains_any(normalized, self.SIMPLE_QUESTION_MARKERS) or normalized.endswith("?"):
            return "simple_question"
        return "new_request"

    def _detect_outcome(self, text: str) -> OutcomePolarity:
        normalized = self._normalize(text)
        if self._contains_any(normalized, self.OUTCOME_WORSE_MARKERS):
            return "worse"
        if self._contains_any(normalized, self.OUTCOME_NO_CHANGE_MARKERS):
            return "no_change"
        if self._contains_any(normalized, self.OUTCOME_IMPROVED_MARKERS):
            if "un poco" in normalized:
                return "partial_relief"
            return "improved"
        return "unknown"

    def _detect_continuity(
        self,
        previous_route: Optional[Domain],
        route_id: Domain,
        turn_family: TurnFamily,
        previous_frame: Dict[str, Any],
    ) -> Tuple[float, Optional[str]]:
        if not previous_route:
            return 0.0, None

        if turn_family in self.FOLLOWUP_FAMILIES:
            return 0.96, "followup_on_active_route"

        if previous_route == route_id:
            return 0.82, "same_route_repeated"

        previous_domain = str(previous_frame.get("conversation_domain") or "").strip()
        current_domain = self.ROUTE_TO_CONVERSATION_DOMAIN.get(route_id)
        if previous_domain and previous_domain == current_domain:
            return 0.74, "same_conversation_domain"

        return 0.22, "new_route"

    def _is_covered(
        self,
        route_id: Domain,
        previous_route: Optional[Domain],
        turn_family: TurnFamily,
        continuity_score: float,
    ) -> bool:
        if route_id in {"crisis", "ansiedad", "bloqueo_ejecutivo", "sueno", "sobrecarga_cuidador", "meta_question"}:
            return True
        if route_id in {"clarificacion", "rechazo_estrategia", "cierre", "pregunta_simple"}:
            return True
        if previous_route and continuity_score >= 0.7 and turn_family in self.FOLLOWUP_FAMILIES:
            return True
        return False

    def _resolve_guidance_mode(
        self,
        turn_family: TurnFamily,
        outcome: OutcomePolarity,
        response_plan: Optional[ResponsePlan],
    ) -> GuidanceMode:
        if response_plan and response_plan.goal == "close_temporarily":
            return "close"
        if response_plan and response_plan.goal == "switch_strategy":
            return "switch"
        if response_plan and response_plan.goal == "decide_one_path":
            return "hold"
        if turn_family in self.DIRECT_FAMILIES:
            return "direct"
        if turn_family == "closure_or_pause":
            return "close"
        if turn_family == "strategy_rejection" or outcome in {"no_change", "worse"}:
            return "switch"
        if turn_family in {"clarification_request", "blocked_followup", "literal_phrase_request"}:
            return "hold"
        if turn_family in {"followup_acceptance", "specific_action_request", "post_action_followup"}:
            return "advance"
        if outcome in {"partial_relief", "improved"} or (response_plan and response_plan.close_softly):
            return "hold"
        return "direct"

    def _next_step_index(
        self,
        previous_state: Dict[str, Any],
        route_id: Domain,
        guidance_mode: GuidanceMode,
        continuity_score: float,
        max_steps: int,
    ) -> int:
        previous_route = previous_state.get("route_id")
        previous_step = int(previous_state.get("step_index", 0) or 0)

        if previous_route != route_id or continuity_score < 0.7:
            return 0
        if guidance_mode == "advance":
            return min(previous_step + 1, max(max_steps - 1, 0))
        return previous_step

    def _response_shape(self, result: SupportFlowResult) -> str:
        plan = result.response_plan
        if not plan:
            return "simple_answer"
        if plan.literal_phrase:
            return "literal_phrase"
        if result.turn_family == "meta_question":
            return "meta_answer"
        if result.turn_family == "simple_question":
            return "simple_answer"
        if result.guidance_mode == "close":
            return "closure_pause"
        if result.guidance_mode == "switch":
            return "strategy_switch"
        if result.turn_family == "clarification_request":
            return "clarify_simple"
        if result.turn_family == "outcome_report":
            return "hold_line" if result.outcome in {"partial_relief", "improved"} else "check_effect"
        if plan.micro_practice:
            return "grounding"
        return "single_action"

    def _selected_microaction(self, response_plan: ResponsePlan) -> Optional[str]:
        if response_plan.literal_phrase:
            return None
        if response_plan.next_step:
            return response_plan.next_step.strip()
        if response_plan.micro_practice:
            return response_plan.micro_practice.strip()
        if response_plan.main_response:
            return response_plan.main_response.strip()
        return None

    def _candidate_actions(self, response_plan: ResponsePlan) -> List[str]:
        candidates: List[str] = []
        if response_plan.next_step:
            candidates.append(response_plan.next_step.strip())
        if response_plan.micro_practice:
            candidates.append(response_plan.micro_practice.strip())
        if response_plan.main_response:
            candidates.append(response_plan.main_response.strip())
        seen: List[str] = []
        for candidate in candidates:
            if candidate and candidate not in seen:
                seen.append(candidate)
        return seen

    def _intervention_level(self, result: SupportFlowResult) -> str:
        if result.route_id == "crisis":
            return "high"
        if result.guidance_mode in {"switch", "advance"}:
            return "medium"
        return "low"

    def _looks_like_outcome_report(self, normalized: str) -> bool:
        return (
            self._contains_any(normalized, self.OUTCOME_IMPROVED_MARKERS)
            or self._contains_any(normalized, self.OUTCOME_WORSE_MARKERS)
            or self._contains_any(normalized, self.OUTCOME_NO_CHANGE_MARKERS)
        )

    def _build_contextual_response_plan(
        self,
        signal: UserSignal,
        route_id: Domain,
        previous_frame: Dict[str, Any],
        action_memory: Dict[str, Any],
    ) -> ResponsePlan:
        normalized = self._normalize(signal.user_text)

        if self._should_clarify_current_action(
            turn_family=signal.turn_family,
            normalized=normalized,
            action_memory=action_memory,
        ):
            return self._build_current_action_clarification_plan(
                route_id=route_id,
                normalized=normalized,
                action_memory=action_memory,
            )

        if signal.turn_family == "blocked_followup":
            blocked_plan = self._build_blocked_followup_plan(
                route_id=route_id,
                normalized=normalized,
                action_memory=action_memory,
            )
            if blocked_plan:
                return blocked_plan

        if signal.turn_family == "post_action_followup":
            post_action_plan = self._build_post_action_followup_plan(
                route_id=route_id,
                normalized=normalized,
                previous_frame=previous_frame,
                action_memory=action_memory,
                outcome=signal.outcome,
            )
            if post_action_plan:
                return post_action_plan

        return build_response_plan(signal)

    def _extract_action_memory(
        self,
        previous_frame: Dict[str, Any],
        fallback_route: Optional[Domain],
    ) -> Dict[str, Any]:
        support_state = dict(previous_frame.get("support_flow_state") or {})
        last_action_domain = (
            previous_frame.get("last_action_domain")
            or support_state.get("last_action_domain")
            or previous_frame.get("conversation_domain")
            or support_state.get("conversation_domain")
            or self.ROUTE_TO_CONVERSATION_DOMAIN.get(fallback_route or "general", "apoyo_general")
        )
        return {
            "last_action_instruction": str(
                previous_frame.get("last_action_instruction")
                or support_state.get("last_action_instruction")
                or ""
            ).strip(),
            "last_action_type": str(
                previous_frame.get("last_action_type")
                or support_state.get("last_action_type")
                or ""
            ).strip(),
            "last_action_goal": str(
                previous_frame.get("last_action_goal")
                or support_state.get("last_action_goal")
                or ""
            ).strip(),
            "last_action_domain": str(last_action_domain or "").strip(),
            "action_followup_count": int(support_state.get("action_followup_count", 0) or 0),
            "recent_followup_modes": list(support_state.get("recent_followup_modes") or []),
        }

    def _has_active_action(self, action_memory: Dict[str, Any]) -> bool:
        return any(
            str(action_memory.get(key) or "").strip()
            for key in ("last_action_instruction", "last_action_type", "last_action_goal")
        )

    def _looks_like_current_action_clarification(self, normalized: str) -> bool:
        if normalized in {"que", "como", "cual", "por donde"}:
            return True
        return self._contains_any(normalized, self.ACTION_CLARIFICATION_MARKERS)

    def _looks_like_direct_action_question(self, normalized: str) -> bool:
        if normalized in {"que", "como", "cual", "por donde"}:
            return True
        return self._contains_any(normalized, self.DIRECT_ACTION_QUESTION_MARKERS)

    def _wants_literal_phrase(self, normalized: str, action_memory: Dict[str, Any]) -> bool:
        if self._contains_any(normalized, ["que frase", "que digo", "que le digo", "cual frase"]):
            return True
        return str(action_memory.get("last_action_type") or "").strip() == "literal_phrase"

    def _is_post_action_followup(self, normalized: str) -> bool:
        if normalized in {"ya", "listo", "hecho"}:
            return True
        return self._contains_any(normalized, self.POST_ACTION_MARKERS)

    def _should_clarify_current_action(
        self,
        turn_family: TurnFamily,
        normalized: str,
        action_memory: Dict[str, Any],
    ) -> bool:
        if not self._has_active_action(action_memory):
            return False
        if turn_family in {"clarification_request", "literal_phrase_request"}:
            return True
        if turn_family == "simple_question" and self._looks_like_direct_action_question(normalized):
            return True
        return False

    def _build_current_action_clarification_plan(
        self,
        route_id: Domain,
        normalized: str,
        action_memory: Dict[str, Any],
    ) -> ResponsePlan:
        last_action_type = str(action_memory.get("last_action_type") or "").strip()
        instruction = str(action_memory.get("last_action_instruction") or "").strip()

        if self._wants_literal_phrase(normalized=normalized, action_memory=action_memory):
            literal_phrase = instruction if last_action_type == "literal_phrase" and instruction else self._default_literal_phrase(route_id)
            return ResponsePlan(
                goal="clarify_current_action",
                tone="claro_directo",
                validation="",
                main_response="La frase es esta:",
                literal_phrase=literal_phrase,
                optional_followup=(
                    "Dila tal cual y no agregues otra instruccion al mismo tiempo."
                    if route_id == "crisis"
                    else "Usa esa frase tal cual, sin hacerla mas larga."
                ),
                tags=[route_id, "clarify_current_action", "literal_phrase"],
            )

        if last_action_type == "environment_step":
            return ResponsePlan(
                goal="clarify_current_action",
                tone="claro_directo",
                validation="",
                main_response=self._environment_examples_response(route_id),
                optional_followup="Elige solo una y cambia esa primero.",
                tags=[route_id, "clarify_current_action", "environment_examples"],
            )

        if last_action_type == "grounding_step":
            return ResponsePlan(
                goal="clarify_current_action",
                tone="claro_directo",
                validation="",
                main_response=(
                    "Hazlo literal: pies en el piso, suelta el aire mas largo una vez y mira tres cosas alrededor."
                ),
                tags=[route_id, "clarify_current_action", "grounding"],
            )

        if last_action_type == "sleep_step":
            return ResponsePlan(
                goal="clarify_current_action",
                tone="claro_directo",
                validation="",
                main_response=(
                    "Haz una sola accion real de sueño: baja la luz o la pantalla y deja 5 a 10 minutos sin exigencia."
                ),
                tags=[route_id, "clarify_current_action", "sleep"],
            )

        if last_action_type in {"executive_step", "action_step"} and "linea de que" in normalized:
            return ResponsePlan(
                goal="clarify_current_action",
                tone="claro_directo",
                validation="",
                main_response=(
                    "De una linea minima para abrir la tarea: puede ser el titulo del tema, la consigna o una primera frase obvia."
                ),
                optional_followup=(
                    "Si no sabes cual, escribe solo el nombre de la materia o dime el nombre y partimos desde ahi."
                ),
                tags=[route_id, "clarify_current_action", "line_start"],
            )

        if last_action_type in {"executive_step", "action_step"} and "por donde" in normalized:
            return ResponsePlan(
                goal="clarify_current_action",
                tone="claro_directo",
                validation="",
                main_response=self._starting_point_response(route_id),
                tags=[route_id, "clarify_current_action", "starting_point"],
            )

        if last_action_type in {"executive_step", "action_step"}:
            return ResponsePlan(
                goal="clarify_current_action",
                tone="claro_directo",
                validation="",
                main_response=self._simple_action_explanation(route_id, instruction),
                optional_followup="Haz solo eso primero.",
                tags=[route_id, "clarify_current_action", "action_step"],
            )

        return ResponsePlan(
            goal="clarify_current_action",
            tone="claro_directo",
            validation="",
            main_response=self._simple_action_explanation(route_id, instruction),
            optional_followup="Haz solo eso y luego vemos si hace falta algo mas.",
            tags=[route_id, "clarify_current_action"],
        )

    def _build_blocked_followup_plan(
        self,
        route_id: Domain,
        normalized: str,
        action_memory: Dict[str, Any],
    ) -> Optional[ResponsePlan]:
        del action_memory

        if route_id == "bloqueo_ejecutivo":
            if "no se que toca" in normalized or "no se que sigue" in normalized or "no se que hacer" in normalized:
                return ResponsePlan(
                    goal="lower_demand_for_block",
                    tone="claro_directo",
                    validation="",
                    main_response="Haz solo esto: abre la materia o tarea que mas urge hoy.",
                    optional_followup="Si no sabes cual, dime el nombre de una materia y partimos desde ahi.",
                    tags=["bloqueo", "lower_demand", "direct_answer"],
                )
            return ResponsePlan(
                goal="lower_demand_for_block",
                tone="claro_directo",
                validation="",
                main_response="Haz solo esto: abre una sola materia o tarea. Nada mas.",
                optional_followup="Si no sabes cual elegir, yo te ayudo a escoger la mas urgente.",
                tags=["bloqueo", "lower_demand", "direct_answer"],
            )

        if route_id == "sueno":
            return ResponsePlan(
                goal="one_sleep_step",
                tone="claro_directo",
                validation="",
                main_response="Haz una sola accion real: baja la luz o apaga la pantalla y deja 5 a 10 minutos sin exigencia.",
                tags=["sueno", "one_step", "direct_answer"],
            )

        return None

    def _build_post_action_followup_plan(
        self,
        route_id: Domain,
        normalized: str,
        previous_frame: Dict[str, Any],
        action_memory: Dict[str, Any],
        outcome: OutcomePolarity,
    ) -> Optional[ResponsePlan]:
        previous_state = dict(previous_frame.get("support_flow_state") or {})
        previous_count = int(previous_state.get("action_followup_count", 0) or 0)

        if self._should_force_followup_exit(previous_state=previous_state, action_memory=action_memory):
            return self._build_followup_exit_plan(
                route_id=route_id,
                normalized=normalized,
                outcome=outcome,
            )

        if route_id == "crisis" and previous_count >= 1 and normalized == "ya":
            return ResponsePlan(
                goal="hold_line",
                tone="claro_directo",
                validation="",
                main_response="Bien. Por ahora no agregues otra indicacion: sostén la frase breve y el entorno mas bajo un momento.",
                optional_followup="Si vuelve a subir o no baja nada, ahi si cambiamos una sola cosa o cerramos por ahora.",
                close_softly=True,
                tags=["crisis", "hold", "post_action_followup"],
            )

        if route_id == "ansiedad":
            return ResponsePlan(
                goal="next_distinct_step",
                tone="claro_directo",
                validation="",
                main_response="Ahora elige una sola presion real de hoy y dejala visible en una frase breve.",
                optional_followup="No abras otra cosa todavia.",
                tags=["ansiedad", "next_step", "post_action_followup"],
            )

        if route_id == "bloqueo_ejecutivo":
            return ResponsePlan(
                goal="next_distinct_step",
                tone="claro_directo",
                validation="",
                main_response="Ahora deja una sola salida visible: escribe solo el titulo o una primera linea minima.",
                optional_followup="Si no sabes cual, dime la materia y te doy una.",
                tags=["bloqueo", "next_step", "post_action_followup"],
            )

        if route_id == "sueno":
            return ResponsePlan(
                goal="hold_line",
                tone="claro_directo",
                validation="",
                main_response="Ahora no sumes mas de una medida: sostén la baja de luz o pantalla 5 a 10 minutos.",
                optional_followup="Si sigue igual despues, cambiamos de via sin meter todo junto.",
                tags=["sueno", "hold", "post_action_followup"],
            )

        return None

    def _should_force_followup_exit(
        self,
        previous_state: Dict[str, Any],
        action_memory: Dict[str, Any],
    ) -> bool:
        if not self._has_active_action(action_memory):
            return False
        action_followup_count = int(previous_state.get("action_followup_count", 0) or 0)
        recent_modes = list(previous_state.get("recent_followup_modes") or [])
        if action_followup_count >= self.FOLLOWUP_EXIT_THRESHOLD:
            return True
        return recent_modes[-3:] == ["check_effect", "hold_line", "adjustment"]

    def _build_followup_exit_plan(
        self,
        route_id: Domain,
        normalized: str,
        outcome: OutcomePolarity,
    ) -> ResponsePlan:
        if outcome in {"no_change", "worse"}:
            return ResponsePlan(
                goal="switch_strategy",
                tone="claro_directo",
                validation="",
                main_response=self._switch_strategy_message(route_id),
                tags=[route_id, "followup_exit", "switch"],
            )

        if normalized == "ya" or route_id in {"crisis", "sueno"}:
            return ResponsePlan(
                goal="close_temporarily",
                tone="claro_directo",
                validation="",
                main_response=self._close_temporarily_message(route_id),
                close_softly=True,
                tags=[route_id, "followup_exit", "close"],
            )

        return ResponsePlan(
            goal="decide_one_path",
            tone="claro_directo",
            validation="",
            main_response=self._decide_one_path_message(route_id),
            tags=[route_id, "followup_exit", "decide"],
        )

    def _resolve_action_state(
        self,
        response_plan: Optional[ResponsePlan],
        route_id: Domain,
        conversation_domain: str,
        previous_action: Dict[str, Any],
    ) -> Dict[str, Optional[str]]:
        if route_id in {"meta_question", "pregunta_simple"} and self._has_active_action(previous_action):
            return {
                "last_action_instruction": previous_action.get("last_action_instruction") or None,
                "last_action_type": previous_action.get("last_action_type") or None,
                "last_action_goal": previous_action.get("last_action_goal") or None,
                "last_action_domain": previous_action.get("last_action_domain") or None,
            }
        if not response_plan:
            return {
                "last_action_instruction": previous_action.get("last_action_instruction") or None,
                "last_action_type": previous_action.get("last_action_type") or None,
                "last_action_goal": previous_action.get("last_action_goal") or None,
                "last_action_domain": previous_action.get("last_action_domain") or conversation_domain,
            }

        instruction = self._extract_action_instruction_from_plan(
            response_plan=response_plan,
            previous_action=previous_action,
        )
        action_type = self._infer_action_type_from_plan(
            response_plan=response_plan,
            route_id=route_id,
            instruction=instruction,
            previous_action=previous_action,
        )

        return {
            "last_action_instruction": instruction or previous_action.get("last_action_instruction") or None,
            "last_action_type": action_type or previous_action.get("last_action_type") or None,
            "last_action_goal": response_plan.goal or previous_action.get("last_action_goal") or None,
            "last_action_domain": conversation_domain or previous_action.get("last_action_domain") or None,
        }

    def _extract_action_instruction_from_plan(
        self,
        response_plan: ResponsePlan,
        previous_action: Dict[str, Any],
    ) -> str:
        if response_plan.literal_phrase:
            return response_plan.literal_phrase.strip().rstrip(".")
        if response_plan.next_step:
            return response_plan.next_step.strip().rstrip(".")
        if response_plan.micro_practice and not response_plan.main_response:
            return response_plan.micro_practice.strip().rstrip(".")

        main_response = str(response_plan.main_response or "").strip()
        normalized_main = self._normalize(main_response)
        if not main_response:
            return str(previous_action.get("last_action_instruction") or "").strip()
        if main_response.endswith("?"):
            return str(previous_action.get("last_action_instruction") or "").strip()
        if normalized_main.startswith("ahora mira solo esto"):
            return str(previous_action.get("last_action_instruction") or "").strip()
        return main_response.rstrip(".")

    def _infer_action_type_from_plan(
        self,
        response_plan: ResponsePlan,
        route_id: Domain,
        instruction: str,
        previous_action: Dict[str, Any],
    ) -> Optional[str]:
        text = self._normalize(
            " ".join(
                part
                for part in [
                    response_plan.main_response,
                    response_plan.next_step,
                    response_plan.literal_phrase,
                    response_plan.micro_practice,
                    instruction,
                ]
                if part
            )
        )
        if response_plan.literal_phrase:
            return "literal_phrase"
        if route_id == "sueno":
            return "sleep_step"
        if route_id == "bloqueo_ejecutivo":
            return "executive_step"
        if route_id == "ansiedad" and (
            response_plan.micro_practice
            or self._contains_any(text, ["pies en el piso", "suelta el aire", "mira tres cosas", "nombra tres cosas"])
        ):
            return "grounding_step"
        if self._contains_any(
            text,
            [
                "ruido",
                "gente",
                "exigencia",
                "exigencias",
                "preguntas",
                "luces",
                "contacto",
                "pantalla",
                "estimulo",
                "estimulos",
                "entorno",
            ],
        ):
            return "environment_step"
        return str(previous_action.get("last_action_type") or "").strip() or "action_step"

    def _resolve_followup_trace(
        self,
        previous_state: Dict[str, Any],
        turn_family: TurnFamily,
        route_id: Domain,
        continuity_score: float,
        action_memory: Dict[str, Any],
        response_plan: Optional[ResponsePlan],
        outcome: OutcomePolarity,
        guidance_mode: GuidanceMode,
    ) -> Dict[str, Any]:
        same_route = previous_state.get("route_id") == route_id and continuity_score >= 0.7
        if not same_route:
            previous_count = 0
            previous_modes: List[str] = []
        else:
            previous_count = int(previous_state.get("action_followup_count", 0) or 0)
            previous_modes = list(previous_state.get("recent_followup_modes") or [])

        has_action = self._has_active_action(action_memory)
        mode = self._resolve_followup_mode(
            turn_family=turn_family,
            response_plan=response_plan,
            outcome=outcome,
            guidance_mode=guidance_mode,
        )

        if has_action and turn_family in self.COUNTED_ACTION_FOLLOWUP_FAMILIES:
            action_followup_count = previous_count + 1 if same_route else 1
        elif not same_route:
            action_followup_count = 0
        else:
            action_followup_count = previous_count

        if has_action and turn_family in self.ACTION_FOLLOWUP_FAMILIES and mode:
            recent_followup_modes = (previous_modes + [mode])[-4:]
        elif not same_route:
            recent_followup_modes = []
        else:
            recent_followup_modes = previous_modes[-4:]

        followup_exit = response_plan.goal if response_plan and response_plan.goal in self.FOLLOWUP_EXIT_GOALS else None
        return {
            "action_followup_count": action_followup_count,
            "recent_followup_modes": recent_followup_modes,
            "followup_exit": followup_exit,
        }

    def _resolve_followup_mode(
        self,
        turn_family: TurnFamily,
        response_plan: Optional[ResponsePlan],
        outcome: OutcomePolarity,
        guidance_mode: GuidanceMode,
    ) -> str:
        if response_plan and response_plan.goal in self.FOLLOWUP_EXIT_GOALS:
            return response_plan.goal
        if response_plan and response_plan.goal in {"clarify_current_action", "hold_line", "next_distinct_step"}:
            return response_plan.goal
        if turn_family in {"clarification_request", "literal_phrase_request"}:
            return "clarify_current_action"
        if turn_family == "blocked_followup":
            return "adjustment"
        if outcome in {"partial_relief", "improved"} or (response_plan and response_plan.close_softly):
            return "hold_line"
        if turn_family == "post_action_followup":
            return "check_effect"
        if turn_family == "followup_acceptance":
            return "next_distinct_step"
        if guidance_mode == "switch":
            return "switch_strategy"
        return guidance_mode

    def _environment_examples_response(self, route_id: Domain) -> str:
        if route_id == "crisis":
            return "Baja una sola demanda concreta: ruido, gente cerca, preguntas o exigencias, luces o contacto"
        if route_id == "sueno":
            return "Ajusta solo una cosa del entorno: luz, ruido o pantalla"
        return "Ajusta solo una cosa del entorno: ruido, gente o exigencia"

    def _starting_point_response(self, route_id: Domain) -> str:
        if route_id == "bloqueo_ejecutivo":
            return "Empieza por la materia o tarea que mas urge hoy. Si no sabes cual, dime el nombre de una y partimos desde ahi."
        return "Empieza por la accion mas pequena y visible de las que ya tenemos."

    def _simple_action_explanation(self, route_id: Domain, instruction: str) -> str:
        if route_id == "bloqueo_ejecutivo" and not instruction:
            return "Haz una sola cosa: abre la materia o tarea mas urgente de hoy."
        if route_id == "ansiedad" and not instruction:
            return "Haz una sola cosa: pies en el piso y una exhalacion larga."
        if instruction:
            return f"La accion actual es esta: {instruction}"
        return "La accion actual es hacer una sola cosa pequeña y literal."

    def _default_literal_phrase(self, route_id: Domain) -> str:
        if route_id == "crisis":
            return "Estoy aqui contigo. No hace falta hablar mucho ahora. Vamos a bajar esto juntos."
        if route_id == "ansiedad":
            return "Solo una cosa a la vez. Ahora no tengo que resolver todo."
        return "Vamos paso a paso. Solo una cosa ahora."

    def _switch_strategy_message(self, route_id: Domain) -> str:
        if route_id == "crisis":
            return "No vamos a repetir lo mismo. Cambia una sola via: menos palabras y mas espacio seguro, o menos gente y ruido."
        if route_id == "ansiedad":
            return "No seguimos con el mismo carril. Cambiamos a una sola via distinta: cuerpo, entorno o una frase breve."
        if route_id == "bloqueo_ejecutivo":
            return "No vamos a empujar mas el mismo paso. Dime la materia y te doy una sola linea de arranque."
        if route_id == "sueno":
            return "No seguimos intentando lo mismo. Cambia a una sola medida real: menos pantalla o menos luz por unos minutos."
        return "Cambiemos a una sola via distinta, sin abrir mas de un frente."

    def _close_temporarily_message(self, route_id: Domain) -> str:
        if route_id == "crisis":
            return "Por ahora no metas otro paso. Sosten la frase breve y el entorno mas bajo un momento, y cerramos ahi por ahora."
        if route_id == "sueno":
            return "Ya no sumes otra medida. Sosten luz baja o pantalla fuera 5 a 10 minutos y por ahora cerramos ahi."
        return "Por ahora no hace falta meter otro paso. Sosten esta accion un momento y cerramos aqui temporalmente."

    def _decide_one_path_message(self, route_id: Domain) -> str:
        if route_id == "ansiedad":
            return "No vamos a abrir mas pasos ahorita. Elige una sola via: o sostienes esto un minuto y cerramos por ahora, o me dices la presion principal y vemos solo esa."
        if route_id == "bloqueo_ejecutivo":
            return "No metas otro paso ahorita. O dejas solo la materia abierta y cerramos por ahora, o me dices la materia y te doy una sola linea."
        return "Ahora toca una sola decision: o sostener esto por ahora, o cambiar a una sola via distinta."

    def _contains_any(self, normalized_text: str, phrases: List[str]) -> bool:
        return any(phrase in normalized_text for phrase in phrases)

    def _normalize(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKD", str(text or "").strip().lower())
        normalized = "".join(char for char in normalized if not unicodedata.combining(char))
        return " ".join(normalized.split())


if __name__ == "__main__":
    engine = SupportFlowEngine()
    samples = [
        ("Esta ocurriendo una crisis y necesito ayuda", {}),
        ("que le digo", {"last_action_instruction": "Estoy aqui contigo", "last_action_type": "literal_phrase", "conversation_domain": "crisis_activa"}),
        ("no me sirve", {"support_flow_state": {"route_id": "ansiedad"}, "conversation_domain": "ansiedad_cognitiva"}),
        ("y luego", {"support_flow_state": {"route_id": "bloqueo_ejecutivo", "step_index": 0}, "conversation_domain": "disfuncion_ejecutiva"}),
    ]
    for message, previous in samples:
        result = engine.resolve_turn(
            source_message=message,
            previous_frame=previous,
        )
        print("-" * 72)
        print(message)
        print(result.to_dict())
