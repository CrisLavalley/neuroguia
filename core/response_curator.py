from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional


class ResponseCurator:
    """
    Lightweight LLM guardrail.

    It no longer rewrites or reconstructs the answer.
    It only:
    - extracts the raw text
    - cleans internal/system artifacts
    - applies light safety and format validation
    - approves or rejects the response
    """

    INTERNAL_PATTERNS = [
        r"\bapoyo_general\b",
        r"\bgeneral_support\b",
        r"\bdisfuncion_ejecutiva\b",
        r"\bansiedad_cognitiva\b",
        r"\bprevencion_escalada\b",
        r"\bregulacion_post_evento\b",
        r"\bsobrecarga_sensorial\b",
        r"\btransicion_rigidez\b",
        r"\bsueno_regulacion\b",
        r"\bcrisis_activa\b",
        r"\breception_containment\b",
        r"\badaptive_intervention\b",
        r"\bfocus_clarification\b",
        r"\bconversation_phase\b",
        r"\bturn_type\b",
        r"\bclarification_mode\b",
        r"\bcrisis_guided_mode\b",
    ]

    TEMPLATE_MARKERS = [
        "vamos paso a paso",
        "haz solo esto ahora",
        "haz solo esta accion concreta",
        "que parte te serviria mas ordenar primero",
        "voy con algo concreto",
        "vamos con algo breve y manejable",
        "por ahora ayuda mas esto",
        "voy a ir a algo concreto",
        "te sirve mas que elija yo el primer paso contigo",
        "ve con una sola cosa primero",
    ]

    ROBOTIC_OPENING_PATTERNS = [
        r"^\s*la respuesta mas util aqui es[:\s,]*",
        r"^\s*la respuesta mas util es[:\s,]*",
        r"^\s*lo mas util suele ser[:\s,]*",
        r"^\s*lo mas util aqui es[:\s,]*",
        r"^\s*en este caso[:\s,]*",
    ]

    ABSTRACT_MARKERS = {
        "nombrar que el momento esta pesado",
        "nombrar el momento",
        "mantenerlo simple y concreto",
        "la respuesta mas util",
        "lo mas util",
        "dejar solo esto vivo",
        "dejar una sola cosa clara",
        "solo me refiero",
    }

    LOOP_MARKERS = {
        "ya",
        "que mas",
        "y luego",
        "y ahora",
        "y ahora que",
        "que sigue",
        "ok que mas",
    }

    DIRECT_META_MARKERS = {
        "quien eres",
        "para que sirves",
        "como puedo llamarte",
        "como te llamo",
        "puedo hablar contigo",
        "puedo platicar contigo",
    }

    CONFUSION_MARKERS = {
        "no entiendo",
        "no te entiendo",
        "que",
        "como",
    }

    STRONG_BLOCK_MARKERS = {
        "no se como",
        "no lo se como",
        "no puedo",
        "no puedo hacerlo",
        "no puedo sola",
    }

    EMOTIONAL_MARKERS = {
        "me siento",
        "estoy mal",
        "estoy cansada",
        "estoy cansado",
        "estoy rebasada",
        "estoy rebasado",
        "estoy agobiada",
        "estoy agobiado",
        "me da miedo",
        "me siento sola",
        "me siento solo",
        "estoy triste",
        "estoy desesperada",
        "estoy desesperado",
    }

    DISALLOWED_NONANSWER_MARKERS = {
        "vamos despacio",
        "mira solo esto",
        "la respuesta mas util",
        "lo mas util",
    }
    REPETITIVE_TEMPLATE_MARKERS = (
        "apoya los pies",
        "baja la activacion",
        "vamos despacio",
    )
    LIGHT_VALIDATION_MARKERS = (
        "tiene sentido",
        "aqui estoy contigo",
        "aqui estoy",
        "suena pesado",
        "entiendo",
        "vamos despacio",
    )

    FRUSTRATION_MARKERS = {
        "no me ayudas",
        "no me ayudas",
        "no ayuda",
        "no sirve",
        "no me sirve",
        "no veo apoyo",
        "no veo que me apoyes",
        "no veo que me apoyes en nada",
        "no veo que me apoyas",
    }

    OVERWHELM_MARKERS = {
        "me gana",
        "todo me gana",
        "se me junta todo",
        "todo se me junta",
        "no puedo",
        "no puedo con todo",
        "no puedo mas",
        "ya no puedo",
    }

    ACTION_CLARIFICATION_MARKERS = {
        "que frase",
        "cual",
        "cual frase",
        "como",
        "como asi",
        "que digo",
        "que le digo",
        "que le digo ahora",
        "que le puedo decir",
        "que puedo decirle",
        "como lo digo",
        "como le digo",
    }

    def curate(
        self,
        llm_result: Optional[Dict[str, Any]] = None,
        fallback_payload: Optional[Dict[str, Any]] = None,
        decision_payload: Optional[Dict[str, Any]] = None,
        stage_result: Optional[Dict[str, Any]] = None,
        state_analysis: Optional[Dict[str, Any]] = None,
        category_analysis: Optional[Dict[str, Any]] = None,
        intent_analysis: Optional[Dict[str, Any]] = None,
        routine_payload: Optional[Dict[str, Any]] = None,
        conversation_control: Optional[Dict[str, Any]] = None,
        conversation_frame: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        response_package: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        llm_result = llm_result or {}
        fallback_payload = fallback_payload or {}
        decision_payload = decision_payload or {}
        stage_result = stage_result or {}
        state_analysis = state_analysis or {}
        category_analysis = category_analysis or {}
        conversation_control = conversation_control or {}
        conversation_frame = conversation_frame or {}
        chat_history = chat_history or []
        response_package = response_package or {}

        raw_text = self._extract_raw_text(llm_result)
        curated_text = self._clean_text(raw_text)
        curated_text = self._trim_for_stage(
            text=curated_text,
            stage=stage_result.get("stage"),
            primary_state=state_analysis.get("primary_state"),
            category=category_analysis.get("detected_category"),
        )
        curated_text, final_control_notes = self._apply_final_control(
            curated_text=curated_text,
            decision_payload=decision_payload,
            stage_result=stage_result,
            state_analysis=state_analysis,
            category_analysis=category_analysis,
            intent_analysis=intent_analysis or {},
            conversation_control=conversation_control,
            conversation_frame=conversation_frame,
            chat_history=chat_history,
            response_package=response_package,
        )
        curated_text = self._clean_text(curated_text)
        curated_text = self._trim_for_stage(
            text=curated_text,
            stage=stage_result.get("stage"),
            primary_state=state_analysis.get("primary_state"),
            category=category_analysis.get("detected_category"),
        )

        quality_score = self._score_quality(
            curated_text=curated_text,
            llm_confidence_hint=float(llm_result.get("llm_confidence_hint", 0.0) or 0.0),
            stage=stage_result.get("stage"),
            category=category_analysis.get("detected_category"),
            primary_state=state_analysis.get("primary_state"),
            decision_payload=decision_payload,
            conversation_control=conversation_control,
            conversation_frame=conversation_frame,
            chat_history=chat_history,
            response_package=response_package,
        )
        approved = self._approve_response(
            curated_text=curated_text,
            quality_score=quality_score,
            stage=stage_result.get("stage"),
            primary_state=state_analysis.get("primary_state"),
            decision_payload=decision_payload,
            conversation_control=conversation_control,
            conversation_frame=conversation_frame,
            chat_history=chat_history,
        )

        curated_structure = self._extract_structure(curated_text)

        return {
            "approved": approved,
            "quality_score": quality_score,
            "curated_response_text": curated_text,
            "curated_response_structure": curated_structure,
            "source_provider": llm_result.get("provider"),
            "used_stub_fallback": bool(llm_result.get("used_stub_fallback", False)),
            "should_learn_if_good": bool(fallback_payload.get("should_learn_if_good", False) and approved),
            "curation_notes": self._build_curation_notes(
                approved=approved,
                quality_score=quality_score,
                category=category_analysis.get("detected_category"),
                stage=stage_result.get("stage"),
                curated_text=curated_text,
                chat_history=chat_history,
            ) + final_control_notes,
        }

    def _apply_final_control(
        self,
        curated_text: str,
        decision_payload: Dict[str, Any],
        stage_result: Dict[str, Any],
        state_analysis: Dict[str, Any],
        category_analysis: Dict[str, Any],
        intent_analysis: Dict[str, Any],
        conversation_control: Dict[str, Any],
        conversation_frame: Dict[str, Any],
        chat_history: List[Dict[str, Any]],
        response_package: Dict[str, Any],
    ) -> tuple[str, List[str]]:
        text = str(curated_text or "").strip()
        notes: List[str] = []
        user_message = self._current_user_message(conversation_control, chat_history)
        normalized_message = self._normalize(user_message)
        domain = str(category_analysis.get("detected_category") or conversation_control.get("domain") or "")
        is_flow_engine_response = self._is_flow_engine_response(
            response_package=response_package,
            decision_payload=decision_payload,
            stage_result=stage_result,
            conversation_frame=conversation_frame,
        )
        last_action_instruction, last_action_type = self._current_action_context(
            conversation_frame=conversation_frame,
            conversation_control=conversation_control,
        )
        repetitive_template = self._detect_overused_repetitive_template(
            text=text,
            chat_history=chat_history,
        )
        if repetitive_template:
            notes.append(
                f"final_control=blocked_repetitive_template:{repetitive_template.replace(' ', '_')}"
            )
            return (
                self._build_non_repetitive_alternative_response(
                    domain=domain,
                    decision_payload=decision_payload,
                    conversation_control=conversation_control,
                    chat_history=chat_history,
                    blocked_marker=repetitive_template,
                ),
                notes,
            )

        cleaned_text = self._light_cleanup_response(text)
        if cleaned_text != text:
            notes.append("final_control=minor_cleanup")
        text = cleaned_text

        if is_flow_engine_response:
            if self._response_is_empty_or_incoherent(text):
                notes.append("final_control=flow_engine_rebuild_only_if_empty_or_incoherent")
                rebuilt_text = self._build_safe_minimal_recovery_response(
                    turn_intent=self._detect_turn_intent(normalized_message),
                    normalized_message=normalized_message,
                    domain=domain,
                    decision_payload=decision_payload,
                    conversation_control=conversation_control,
                    chat_history=chat_history,
                )
                rebuilt_marker = self._detect_overused_repetitive_template(
                    text=rebuilt_text,
                    chat_history=chat_history,
                )
                if rebuilt_marker:
                    notes.append(
                        f"final_control=blocked_repetitive_template:{rebuilt_marker.replace(' ', '_')}"
                    )
                    rebuilt_text = self._build_non_repetitive_alternative_response(
                        domain=domain,
                        decision_payload=decision_payload,
                        conversation_control=conversation_control,
                        chat_history=chat_history,
                        blocked_marker=rebuilt_marker,
                    )
                return rebuilt_text, notes
            notes.append("final_control=respect_flow_engine_response")
            return text, notes

        if not self._response_is_empty_or_incoherent(text):
            notes.append("final_control=light_pass")
            return text, notes

        if self._is_action_clarification_request(
            normalized_message=normalized_message,
            last_action_instruction=last_action_instruction,
        ):
            notes.append("final_control=clarify_current_action")
            return (
                self._build_action_clarification_response(
                    last_action_instruction=last_action_instruction,
                    last_action_type=last_action_type,
                    chat_history=chat_history,
                ),
                notes,
            )

        emotional_priority_reason = self._detect_emotional_priority_need(
            normalized_message=normalized_message,
            chat_history=chat_history,
        )
        if emotional_priority_reason:
            notes.append(f"final_control=emotional_first_{emotional_priority_reason}")
            return (
                self._build_emotional_priority_response(
                    reason=emotional_priority_reason,
                    domain=domain,
                    decision_payload=decision_payload,
                    chat_history=chat_history,
                ),
                notes,
            )

        turn_intent = self._detect_turn_intent(normalized_message)

        if turn_intent == "other" and self._is_loop_followup(normalized_message, conversation_control, chat_history):
            notes.append("final_control=loop_exit")
            text = self._build_loop_exit_response(
                domain=domain,
                decision_payload=decision_payload,
                conversation_control=conversation_control,
                chat_history=chat_history,
            )

        if not text or self._contains_disallowed_nonanswer_phrase(text):
            notes.append("final_control=nonanswer_or_empty")
            text = self._build_priority_response(
                turn_intent=turn_intent,
                normalized_message=normalized_message,
                domain=domain,
                decision_payload=decision_payload,
                conversation_control=conversation_control,
                chat_history=chat_history,
            )
        elif self._contains_abstract_language(text):
            notes.append("final_control=abstract_to_concrete")
            text = self._rewrite_abstract_response(
                turn_intent=turn_intent,
                normalized_message=normalized_message,
                domain=domain,
                decision_payload=decision_payload,
                current_text=text,
                conversation_control=conversation_control,
                chat_history=chat_history,
            )
        elif self._is_similar_to_recent_responses(text, chat_history):
            notes.append("final_control=anti_repetition")
            text = self._build_alternative_response(
                turn_intent=turn_intent,
                normalized_message=normalized_message,
                domain=domain,
                decision_payload=decision_payload,
                current_text=text,
                conversation_control=conversation_control,
                chat_history=chat_history,
            )
        else:
            notes.append(f"final_control=replace_for_{turn_intent}")
            text = self._build_priority_response(
                turn_intent=turn_intent,
                normalized_message=normalized_message,
                domain=domain,
                decision_payload=decision_payload,
                conversation_control=conversation_control,
                chat_history=chat_history,
            )

        repetitive_template = self._detect_overused_repetitive_template(
            text=text,
            chat_history=chat_history,
        )
        if repetitive_template:
            notes.append(
                f"final_control=blocked_repetitive_template:{repetitive_template.replace(' ', '_')}"
            )
            text = self._build_non_repetitive_alternative_response(
                domain=domain,
                decision_payload=decision_payload,
                conversation_control=conversation_control,
                chat_history=chat_history,
                blocked_marker=repetitive_template,
            )

        return text, notes

    def _extract_raw_text(self, llm_result: Dict[str, Any]) -> str:
        response_text = str(llm_result.get("response_text") or "").strip()
        if response_text:
            return response_text

        structure = llm_result.get("response_structure", {}) or {}
        candidates = [
            structure.get("full_text"),
            structure.get("main_guidance"),
            structure.get("opening_validation"),
        ]
        return "\n\n".join([str(candidate).strip() for candidate in candidates if str(candidate or "").strip()])

    def _clean_text(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        for pattern in self.INTERNAL_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        for pattern in self.ROBOTIC_OPENING_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)

        text = re.sub(r"\s+([,.;:])", r"\1", text)
        text = re.sub(r"([,.;:])([^\s])", r"\1 \2", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\.(\s*\.)+", ".", text)
        text = text.strip(" \n\t")

        if text and not re.search(r"[.!?]$", text):
            text += "."

        return text

    def _trim_for_stage(
        self,
        text: str,
        stage: Optional[str],
        primary_state: Optional[str],
        category: Optional[str],
    ) -> str:
        if not text:
            return text

        max_chars = 700
        if stage == "focus_clarification":
            max_chars = 360
        if category == "crisis_activa" or primary_state in {"meltdown", "shutdown"}:
            max_chars = 280

        if len(text) <= max_chars:
            return text

        trimmed = text[:max_chars].rstrip()
        last_break = max(trimmed.rfind(". "), trimmed.rfind("\n"))
        if last_break > 80:
            trimmed = trimmed[: last_break + 1].rstrip()
        if trimmed and not re.search(r"[.!?]$", trimmed):
            trimmed += "."
        return trimmed

    def _extract_structure(self, text: str) -> Dict[str, Any]:
        if not text:
            return {
                "opening_validation": None,
                "main_guidance": None,
                "microaction": None,
                "followup_bridge": None,
                "full_text": "",
            }

        paragraphs = [part.strip() for part in re.split(r"\n\n+", text) if part.strip()]
        if len(paragraphs) >= 3:
            opening = paragraphs[0]
            main_guidance = "\n\n".join(paragraphs[1:-1]).strip()
            followup = paragraphs[-1]
        elif len(paragraphs) == 2:
            opening = paragraphs[0]
            main_guidance = paragraphs[1]
            followup = None
        else:
            sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
            opening = sentences[0] if sentences else text
            main_guidance = " ".join(sentences[1:]).strip() if len(sentences) > 1 else opening
            followup = None

        return {
            "opening_validation": opening or None,
            "main_guidance": main_guidance or None,
            "microaction": None,
            "followup_bridge": followup,
            "full_text": text,
        }

    def _score_quality(
        self,
        curated_text: str,
        llm_confidence_hint: float,
        stage: Optional[str],
        category: Optional[str],
        primary_state: Optional[str],
        decision_payload: Dict[str, Any],
        conversation_control: Dict[str, Any],
        conversation_frame: Dict[str, Any],
        chat_history: List[Dict[str, Any]],
        response_package: Dict[str, Any],
    ) -> float:
        if not curated_text:
            return 0.0

        response_goal = decision_payload.get("response_goal", {}) or {}
        response_shape = str(response_goal.get("response_shape") or "")
        turn_family = str(conversation_control.get("turn_family") or "")
        score = 0.35
        text_norm = self._normalize(curated_text)
        user_message_norm = self._normalize(self._current_user_message(conversation_control, chat_history))
        turn_intent = self._detect_turn_intent(user_message_norm)
        is_flow_engine_response = self._is_flow_engine_response(
            response_package=response_package,
            decision_payload=decision_payload,
            stage_result={"stage": stage},
            conversation_frame=conversation_frame,
        )
        last_action_instruction, last_action_type = self._current_action_context(
            conversation_frame=conversation_frame,
            conversation_control=conversation_control,
        )
        action_clarification_requested = self._is_action_clarification_request(
            normalized_message=user_message_norm,
            last_action_instruction=last_action_instruction,
        )
        clarifies_current_action = self._response_clarifies_action(
            text=curated_text,
            last_action_instruction=last_action_instruction,
            last_action_type=last_action_type,
        )

        if len(curated_text) >= 60:
            score += 0.14
        if len(curated_text) >= 110:
            score += 0.10
        if 20 <= len(curated_text) < 60 and stage in {"focus_clarification", "reception_containment"}:
            score += 0.08
        if stage == "focus_clarification" and len(curated_text) <= 320:
            score += 0.07
        if category == "crisis_activa" and len(curated_text) <= 260:
            score += 0.07
        if primary_state in {"meltdown", "shutdown"} and len(curated_text) <= 260:
            score += 0.06
        if response_shape in {"closure_pause", "permission_pause", "grounding", "load_relief", "sleep_settle", "crisis_containment"} and 18 <= len(curated_text) <= 140:
            score += 0.10
        if turn_family in {"clarification_request", "closure_or_pause", "literal_phrase_request", "specific_action_request"} and len(curated_text) <= 180:
            score += 0.08
        if any(token in text_norm for token in ["puedes decirle", "\"", "solo me refiero", "lo primero es", "hoy basta con"]):
            score += 0.08
        if any(token in text_norm for token in ["voy a dejarlo asi", "voy a elegir", "sin darle mas vueltas", "haz esto en este orden"]):
            score += 0.07
        if self._is_clear_question(user_message_norm) and self._answers_clearly(curated_text, user_message_norm):
            score += 0.14
        if turn_intent == "confusion" and self._response_resolves_confusion(curated_text):
            score += 0.16
        if turn_intent == "strong_block" and self._response_handles_strong_block(curated_text):
            score += 0.16
        if turn_intent == "emotional" and self._response_validates_emotion(curated_text):
            score += 0.14
        if self._is_simple_human_message(user_message_norm) and self._is_natural_human_reply(curated_text, user_message_norm):
            score += 0.14
        if action_clarification_requested and clarifies_current_action:
            score += 0.20
        if 24 <= len(curated_text) <= 220:
            score += 0.08
        if self._response_is_action_like(curated_text):
            score += 0.12
        if self._is_domain_specific_response(
            text=curated_text,
            category=category,
            conversation_frame=conversation_frame,
            decision_payload=decision_payload,
        ):
            score += 0.16
        if is_flow_engine_response:
            score += 0.08
        if "1." in curated_text and "2." in curated_text:
            score += 0.06
        if any(token in text_norm for token in ["no tienes que", "no necesitas decidirlo", "no hace falta decidir"]):
            score += 0.06
        if self._has_template_marker(text_norm):
            score -= 0.24
        if any(token in text_norm for token in self.REPETITIVE_TEMPLATE_MARKERS):
            score -= 0.14
        if self._detect_overused_repetitive_template(curated_text, chat_history):
            score -= 0.30
        if self._is_validation_only_response(curated_text):
            score -= 0.18
        if any(token in text_norm for token in ["vamos despacio", "aqui estoy contigo", "aqui estoy"]) and not self._response_is_action_like(curated_text):
            score -= 0.12
        if text_norm.count("si quieres") > 2:
            score -= 0.08
        if not (action_clarification_requested and clarifies_current_action) and self._looks_recycled_against_history(curated_text, chat_history):
            score -= 0.20
        if not (action_clarification_requested and clarifies_current_action) and self._has_repeated_opening(curated_text, chat_history):
            score -= 0.12
        if self._contains_disallowed_nonanswer_phrase(curated_text) and not self._responds_to_turn_intent(
            text=curated_text,
            turn_intent=turn_intent,
            normalized_message=user_message_norm,
        ):
            score -= 0.20
        if any(re.search(pattern, curated_text, flags=re.IGNORECASE) for pattern in self.INTERNAL_PATTERNS):
            score -= 0.20

        score += min(max(llm_confidence_hint, 0.0), 1.0) * 0.08
        return round(max(min(score, 1.0), 0.0), 4)

    def _approve_response(
        self,
        curated_text: str,
        quality_score: float,
        stage: Optional[str],
        primary_state: Optional[str],
        decision_payload: Dict[str, Any],
        conversation_control: Dict[str, Any],
        conversation_frame: Dict[str, Any],
        chat_history: List[Dict[str, Any]],
    ) -> bool:
        user_message_norm = self._normalize(
            self._current_user_message(conversation_control, chat_history)
        )
        turn_intent = self._detect_turn_intent(user_message_norm)
        last_action_instruction, last_action_type = self._current_action_context(
            conversation_frame=conversation_frame,
            conversation_control=conversation_control,
        )
        action_clarification_requested = self._is_action_clarification_request(
            normalized_message=user_message_norm,
            last_action_instruction=last_action_instruction,
        )
        clarifies_current_action = self._response_clarifies_action(
            text=curated_text,
            last_action_instruction=last_action_instruction,
            last_action_type=last_action_type,
        )
        if not curated_text or len(curated_text.strip()) < 8:
            return False
        if self._has_template_marker(self._normalize(curated_text)):
            return False
        if not (action_clarification_requested and clarifies_current_action) and self._looks_recycled_against_history(curated_text, chat_history):
            return False
        if self._contains_disallowed_nonanswer_phrase(curated_text) and not self._responds_to_turn_intent(
            text=curated_text,
            turn_intent=turn_intent,
            normalized_message=user_message_norm,
        ):
            return False
        if action_clarification_requested and clarifies_current_action and quality_score >= 0.32:
            return True
        if self._is_clear_question(user_message_norm) and self._answers_clearly(curated_text, user_message_norm) and quality_score >= 0.32:
            return True
        if turn_intent == "confusion" and self._response_resolves_confusion(curated_text) and quality_score >= 0.32:
            return True
        if turn_intent == "strong_block" and self._response_handles_strong_block(curated_text) and quality_score >= 0.32:
            return True
        if turn_intent == "emotional" and self._response_validates_emotion(curated_text) and quality_score >= 0.32:
            return True
        if self._is_simple_human_message(user_message_norm) and self._is_natural_human_reply(curated_text, user_message_norm) and quality_score >= 0.32:
            return True
        if quality_score < 0.38:
            return False
        if stage == "focus_clarification" and quality_score < 0.46:
            return False
        if primary_state in {"meltdown", "shutdown"} and quality_score < 0.48:
            return False
        return True

    def _build_curation_notes(
        self,
        approved: bool,
        quality_score: float,
        category: Optional[str],
        stage: Optional[str],
        curated_text: str,
        chat_history: List[Dict[str, Any]],
    ) -> List[str]:
        notes = [f"approved={approved}", f"quality_score={quality_score}"]
        if category:
            notes.append(f"category={category}")
        if stage:
            notes.append(f"stage={stage}")
        if self._has_template_marker(self._normalize(curated_text)):
            notes.append("template_detected=true")
        if self._looks_recycled_against_history(curated_text, chat_history):
            notes.append("recycled_detected=true")
        return notes

    def _is_flow_engine_response(
        self,
        response_package: Dict[str, Any],
        decision_payload: Dict[str, Any],
        stage_result: Dict[str, Any],
        conversation_frame: Dict[str, Any],
    ) -> bool:
        if bool(response_package.get("is_flow_engine_response")):
            return True
        response_metadata = dict(response_package.get("response_metadata", {}) or {})
        if bool(response_metadata.get("is_flow_engine_response")):
            return True
        if str(response_metadata.get("source") or "").strip() == "support_flow_engine":
            return True
        if str(decision_payload.get("decision_mode") or "").strip() == "support_flow_engine":
            return True
        if str(stage_result.get("stage") or "").strip() == "guided_support_flow":
            return True
        support_state = dict(conversation_frame.get("support_flow_state", {}) or {})
        return str(support_state.get("handled_by") or "").strip() == "support_flow_engine"

    def _light_cleanup_response(self, text: str) -> str:
        cleaned = str(text or "").strip()
        if not cleaned:
            return ""
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip()]
        deduped: List[str] = []
        seen: set[str] = set()
        for sentence in sentences:
            normalized = self._normalize(sentence)
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped.append(sentence)
        if deduped:
            cleaned = " ".join(deduped)
        cleaned = re.sub(r"\b(\w+(?:\s+\w+){2,8})\s+\1\b", r"\1", cleaned, flags=re.IGNORECASE)
        return cleaned.strip()

    def _response_is_empty_or_incoherent(self, text: str) -> bool:
        normalized = self._normalize(text)
        if not normalized or len(normalized) < 8:
            return True
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in self.INTERNAL_PATTERNS):
            return True
        if self._contains_disallowed_nonanswer_phrase(text) and not self._response_is_action_like(text):
            return True
        if self._contains_abstract_language(text) and not self._response_is_action_like(text):
            return True
        return False

    def _detect_overused_repetitive_template(
        self,
        text: str,
        chat_history: List[Dict[str, Any]],
    ) -> Optional[str]:
        normalized = self._normalize(text)
        if not normalized:
            return None
        recent_assistant_texts = [
            self._normalize(item)
            for item in self._recent_assistant_texts(chat_history, limit=3)
            if str(item).strip()
        ]
        for marker in self.REPETITIVE_TEMPLATE_MARKERS:
            if marker not in normalized:
                continue
            count = sum(1 for previous in recent_assistant_texts if marker in previous)
            if count >= 2:
                return marker
        return None

    def _build_non_repetitive_alternative_response(
        self,
        domain: str,
        decision_payload: Dict[str, Any],
        conversation_control: Dict[str, Any],
        chat_history: List[Dict[str, Any]],
        blocked_marker: str,
    ) -> str:
        del blocked_marker
        action = self._default_non_repetitive_action(domain, decision_payload)
        turn_family = str(conversation_control.get("turn_family") or "").strip()
        if turn_family == "followup_acceptance":
            return self._pick_non_repeated_variant(
                [
                    f"Lo siguiente es esto: {action}.",
                    f"Sigue por aqui: {action}.",
                ],
                chat_history,
            )
        return self._pick_non_repeated_variant(
            [
                f"Haz esto: {action}.",
                f"Vamos con algo mas concreto: {action}.",
            ],
            chat_history,
        )

    def _build_safe_minimal_recovery_response(
        self,
        turn_intent: str,
        normalized_message: str,
        domain: str,
        decision_payload: Dict[str, Any],
        conversation_control: Dict[str, Any],
        chat_history: List[Dict[str, Any]],
    ) -> str:
        if turn_intent != "other":
            return self._build_priority_response(
                turn_intent=turn_intent,
                normalized_message=normalized_message,
                domain=domain,
                decision_payload=decision_payload,
                conversation_control=conversation_control,
                chat_history=chat_history,
            )
        return self._build_non_repetitive_alternative_response(
            domain=domain,
            decision_payload=decision_payload,
            conversation_control=conversation_control,
            chat_history=chat_history,
            blocked_marker="",
        )

    def _default_non_repetitive_action(
        self,
        domain: str,
        decision_payload: Dict[str, Any],
    ) -> str:
        selected = self._default_concrete_action(domain, decision_payload)
        if not any(marker in self._normalize(selected) for marker in self.REPETITIVE_TEMPLATE_MARKERS):
            return selected
        alternatives = {
            "ansiedad_cognitiva": "nombra una sola presion real de hoy en una frase breve y deja lo demas quieto",
            "disfuncion_ejecutiva": "abre solo el archivo o material que toca",
            "crisis_activa": "baja una fuente de ruido o gente cerca y usa pocas palabras",
            "sueno_regulacion": "baja una sola fuente de estimulo como luz, ruido o pantalla",
            "sobrecarga_cuidador": "suelta una sola carga concreta por ahora: una decision, una tarea o una exigencia",
        }
        return alternatives.get(domain, "haz una sola cosa pequena y visible")

    def _is_validation_only_response(self, text: str) -> bool:
        normalized = self._normalize(text)
        if not normalized:
            return True
        if self._response_is_action_like(text):
            return False
        if len(normalized.split()) > 16:
            return False
        return any(marker in normalized for marker in self.LIGHT_VALIDATION_MARKERS)

    def _is_domain_specific_response(
        self,
        text: str,
        category: Optional[str],
        conversation_frame: Dict[str, Any],
        decision_payload: Dict[str, Any],
    ) -> bool:
        normalized = self._normalize(text)
        if not normalized:
            return False
        response_goal = dict(decision_payload.get("response_goal", {}) or {})
        selected_microaction = self._normalize(
            str(
                decision_payload.get("selected_microaction")
                or response_goal.get("selected_microaction")
                or ""
            )
        )
        if selected_microaction and selected_microaction in normalized:
            return True
        domain = str(category or conversation_frame.get("conversation_domain") or "").strip()
        domain_markers = {
            "ansiedad_cognitiva": ("presion real", "una frase breve", "deja lo demas quieto"),
            "disfuncion_ejecutiva": ("archivo", "material", "titulo", "primera linea", "materia"),
            "crisis_activa": ("ruido", "pocas palabras", "gente cerca", "espacio seguro"),
            "sueno_regulacion": ("estimulo", "pantalla", "luz", "ruido", "dormir"),
            "sobrecarga_cuidador": ("carga concreta", "decision", "tarea", "exigencia", "mi hijo", "mi hija"),
        }
        return any(marker in normalized for marker in domain_markers.get(domain, ()))

    def _normalize(self, text: Optional[str]) -> str:
        normalized = " ".join((text or "").strip().lower().split())
        normalized = unicodedata.normalize("NFKD", normalized)
        normalized = "".join(char for char in normalized if not unicodedata.combining(char))
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        return " ".join(normalized.split())

    def _current_user_message(self, conversation_control: Dict[str, Any], chat_history: List[Dict[str, Any]]) -> str:
        source_message = str(
            conversation_control.get("source_message")
            or conversation_control.get("effective_message")
            or ""
        ).strip()
        if source_message:
            return source_message
        for turn in reversed(chat_history or []):
            if not isinstance(turn, dict):
                continue
            user_text = str(turn.get("user") or "").strip()
            if user_text:
                return user_text
        return ""

    def _current_action_context(
        self,
        conversation_frame: Dict[str, Any],
        conversation_control: Dict[str, Any],
    ) -> tuple[str, Optional[str]]:
        last_action_instruction = str(
            conversation_frame.get("last_action_instruction")
            or conversation_control.get("last_action_instruction")
            or ""
        ).strip()
        last_action_type = str(
            conversation_frame.get("last_action_type")
            or conversation_control.get("last_action_type")
            or ""
        ).strip() or None
        return last_action_instruction, last_action_type

    def _detect_emotional_priority_need(
        self,
        normalized_message: str,
        chat_history: List[Dict[str, Any]],
    ) -> Optional[str]:
        if not normalized_message:
            return None
        if any(marker in normalized_message for marker in self.FRUSTRATION_MARKERS):
            return "frustration"
        if normalized_message in {"no entiendo", "no te entiendo"} or "no te entiendo" in normalized_message:
            return "incomprehension"
        if any(marker in normalized_message for marker in self.OVERWHELM_MARKERS):
            return "overwhelm"
        if self._has_emotional_repetition(normalized_message, chat_history):
            return "emotional_repetition"
        return None

    def _has_emotional_repetition(
        self,
        normalized_message: str,
        chat_history: List[Dict[str, Any]],
    ) -> bool:
        if not self._message_has_emotional_load(normalized_message):
            return False
        recent_user_turns = [
            self._normalize(str(turn.get("user") or ""))
            for turn in (chat_history or [])[-3:]
            if isinstance(turn, dict) and str(turn.get("user") or "").strip()
        ]
        emotional_recent = [
            item
            for item in recent_user_turns
            if self._message_has_emotional_load(item)
        ]
        return len(emotional_recent) >= 1

    def _message_has_emotional_load(self, normalized_message: str) -> bool:
        if not normalized_message:
            return False
        marker_groups = (
            self.FRUSTRATION_MARKERS,
            self.OVERWHELM_MARKERS,
            self.EMOTIONAL_MARKERS,
        )
        return any(
            marker in normalized_message
            for group in marker_groups
            for marker in group
        ) or normalized_message in {"no entiendo", "no te entiendo"}

    def _is_action_clarification_request(
        self,
        normalized_message: str,
        last_action_instruction: str,
    ) -> bool:
        if not last_action_instruction:
            return False
        if normalized_message in self.ACTION_CLARIFICATION_MARKERS:
            return True
        if any(
            marker in normalized_message
            for marker in (
                "que frase",
                "cual frase",
                "que digo",
                "que le digo",
                "que le digo ahora",
                "que le puedo decir",
                "que puedo decirle",
                "como lo digo",
                "como le digo",
            )
        ):
            return True
        words = normalized_message.split()
        if words and len(words) <= 2 and words[0] in {"como", "cual"}:
            return True
        return False

    def _build_action_clarification_response(
        self,
        last_action_instruction: str,
        last_action_type: Optional[str],
        chat_history: List[Dict[str, Any]],
    ) -> str:
        instruction = self._strip_surrounding_quotes(last_action_instruction).strip()
        if last_action_type == "literal_phrase":
            return self._pick_non_repeated_variant(
                [
                    f'La frase concreta es esta: "{instruction}".',
                    f'Puedes decirle esto: "{instruction}".',
                ],
                chat_history,
            )
        return self._pick_non_repeated_variant(
            [
                f"Me refiero a este paso: {instruction}. Haz solo eso por ahora.",
                f"El paso es este: {instruction}. No hace falta abrir otra cosa todavia.",
            ],
            chat_history,
        )

    def _response_clarifies_action(
        self,
        text: str,
        last_action_instruction: str,
        last_action_type: Optional[str],
    ) -> bool:
        instruction_norm = self._normalize(self._strip_surrounding_quotes(last_action_instruction))
        text_norm = self._normalize(text)
        if not instruction_norm or not text_norm:
            return False
        if instruction_norm in text_norm:
            return True

        instruction_tokens = [token for token in instruction_norm.split() if len(token) > 2]
        if not instruction_tokens:
            return False
        text_tokens = set(text_norm.split())
        overlap = sum(1 for token in set(instruction_tokens) if token in text_tokens)
        threshold = 0.6 if len(set(instruction_tokens)) >= 3 else 1.0
        if overlap / max(len(set(instruction_tokens)), 1) >= threshold:
            return True
        if last_action_type == "literal_phrase":
            return any(marker in text_norm for marker in {"frase concreta", "puedes decirle", "me refiero a esta frase"})
        return any(marker in text_norm for marker in {"me refiero a este paso", "el paso es este", "haz solo eso"})

    def _strip_surrounding_quotes(self, text: str) -> str:
        return str(text or "").strip().strip('"').rstrip(".")

    def _contains_abstract_language(self, text: str) -> bool:
        normalized = self._normalize(text)
        return any(marker in normalized for marker in self.ABSTRACT_MARKERS)

    def _contains_disallowed_nonanswer_phrase(self, text: str) -> bool:
        normalized = self._normalize(text)
        return any(marker in normalized for marker in self.DISALLOWED_NONANSWER_MARKERS)

    def _detect_turn_intent(self, normalized_message: str) -> str:
        if self._is_direct_question(normalized_message):
            return "direct_question"
        if self._is_confusion_message(normalized_message):
            return "confusion"
        if self._is_strong_block_message(normalized_message):
            return "strong_block"
        if self._is_emotional_message(normalized_message):
            return "emotional"
        return "other"

    def _responds_to_turn_intent(
        self,
        text: str,
        turn_intent: str,
        normalized_message: str,
    ) -> bool:
        if turn_intent == "direct_question":
            return self._answers_clearly(text, normalized_message)
        if turn_intent == "confusion":
            return self._response_resolves_confusion(text)
        if turn_intent == "strong_block":
            return self._response_handles_strong_block(text)
        if turn_intent == "emotional":
            return self._response_validates_emotion(text)
        return True

    def _build_priority_response(
        self,
        turn_intent: str,
        normalized_message: str,
        domain: str,
        decision_payload: Dict[str, Any],
        conversation_control: Dict[str, Any],
        chat_history: List[Dict[str, Any]],
    ) -> str:
        if turn_intent == "direct_question":
            return self._build_direct_answer(
                normalized_message=normalized_message,
                domain=domain,
                decision_payload=decision_payload,
                conversation_control=conversation_control,
                chat_history=chat_history,
            )
        if turn_intent == "confusion":
            return self._build_confusion_response(
                domain=domain,
                decision_payload=decision_payload,
                chat_history=chat_history,
            )
        if turn_intent == "strong_block":
            return self._build_strong_block_response(
                domain=domain,
                decision_payload=decision_payload,
                chat_history=chat_history,
            )
        if turn_intent == "emotional":
            return self._build_emotional_response(
                normalized_message=normalized_message,
                chat_history=chat_history,
            )
        return self._build_general_human_response(domain=domain, decision_payload=decision_payload, chat_history=chat_history)

    def _is_direct_question(self, normalized_message: str) -> bool:
        if not normalized_message:
            return False
        if any(marker in normalized_message for marker in self.DIRECT_META_MARKERS):
            return True
        if any(marker in normalized_message for marker in {"y despues", "y luego", "que sigue", "y ahora", "y ahora que"}):
            return True
        if "puedo" in normalized_message and "contigo" in normalized_message and (
            "hablar" in normalized_message or "platicar" in normalized_message
        ):
            return True
        if "hago" in normalized_message:
            return True
        if "empiezo" in normalized_message or "comienzo" in normalized_message:
            return True
        return False

    def _is_clear_question(self, normalized_message: str) -> bool:
        return self._is_direct_question(normalized_message)

    def _is_confusion_message(self, normalized_message: str) -> bool:
        if not normalized_message:
            return False
        if normalized_message in self.CONFUSION_MARKERS:
            return True
        return any(marker in normalized_message for marker in {"no entiendo", "no te entiendo"})

    def _is_strong_block_message(self, normalized_message: str) -> bool:
        if not normalized_message:
            return False
        if any(marker in normalized_message for marker in self.STRONG_BLOCK_MARKERS):
            return True
        return normalized_message.startswith("no puedo") or "no se como" in normalized_message

    def _is_emotional_message(self, normalized_message: str) -> bool:
        if not normalized_message:
            return False
        return self._message_has_emotional_load(normalized_message)

    def _is_simple_human_message(self, normalized_message: str) -> bool:
        if not normalized_message:
            return False
        if normalized_message in {"no entiendo", "no te entiendo"}:
            return True
        if normalized_message in {"no se", "no lo se", "no s", "no lo s"}:
            return True
        return "puedo" in normalized_message and "contigo" in normalized_message and (
            "hablar" in normalized_message or "platicar" in normalized_message
        )

    def _is_loop_followup(
        self,
        normalized_message: str,
        conversation_control: Dict[str, Any],
        chat_history: List[Dict[str, Any]],
    ) -> bool:
        if normalized_message in self.LOOP_MARKERS:
            return True
        signals = conversation_control.get("progression_signals", {}) or {}
        if signals.get("repeated_post_action_followup"):
            return True
        turn_family = str(conversation_control.get("turn_family") or "")
        if turn_family != "post_action_followup":
            return False
        recent_user_turns = [
            self._normalize(str(turn.get("user") or ""))
            for turn in (chat_history or [])[-2:]
            if isinstance(turn, dict) and str(turn.get("user") or "").strip()
        ]
        recent_user_turns.append(normalized_message)
        return sum(1 for item in recent_user_turns if item in self.LOOP_MARKERS) >= 2

    def _answers_clearly(self, text: str, normalized_message: str) -> bool:
        normalized_text = self._normalize(text)
        if not normalized_text:
            return False
        if "quien eres" in normalized_message:
            return "soy neuroguia" in normalized_text or normalized_text.startswith("soy ")
        if "para que sirves" in normalized_message:
            return "sirvo para" in normalized_text or "puedo ayudarte" in normalized_text
        if "como puedo llamarte" in normalized_message or "como te llamo" in normalized_message:
            return any(
                token in normalized_text
                for token in {"puedes decirme", "llamame", "soy neuroguia"}
            )
        if "puedo" in normalized_message and "contigo" in normalized_message and (
            "hablar" in normalized_message or "platicar" in normalized_message
        ):
            return (
                normalized_text.startswith("si")
                or normalized_text.startswith("claro")
                or "aqui estoy contigo" in normalized_text
                or "aqui estoy" in normalized_text
            )
        if any(token in normalized_message for token in {"y despues", "y luego", "que sigue", "y ahora", "y ahora que"}):
            return any(
                token in normalized_text
                for token in {
                    "despues va esto",
                    "lo siguiente es",
                    "por ahora basta",
                    "quedate solo con eso",
                    "haz esto",
                    "empieza por",
                }
            )
        if "hago" in normalized_message or "empiezo" in normalized_message or "comienzo" in normalized_message:
            return any(
                token in normalized_text
                for token in {"haz esto", "empieza por", "abre", "apoya", "baja", "quedate con"}
            )
        return True

    def _is_natural_human_reply(self, text: str, normalized_message: str) -> bool:
        normalized_text = self._normalize(text)
        if not normalized_text:
            return False
        if normalized_message in {"no entiendo", "no te entiendo"}:
            return any(token in normalized_text for token in {"te lo digo mas simple", "te lo digo simple", "esta bien"})
        if normalized_message in {"no se", "no lo se", "no s", "no lo s"}:
            return any(token in normalized_text for token in {"esta bien", "aqui estoy", "no hay prisa"})
        if "puedo" in normalized_message and "contigo" in normalized_message and (
            "hablar" in normalized_message or "platicar" in normalized_message
        ):
            return self._answers_clearly(text, normalized_message)
        return True

    def _response_resolves_confusion(self, text: str) -> bool:
        normalized_text = self._normalize(text)
        if not normalized_text:
            return False
        has_simple_frame = any(
            token in normalized_text
            for token in {"solo haz esto", "haz una sola cosa", "te lo digo mas simple", "te lo digo simple"}
        )
        has_close = any(token in normalized_text for token in {"nada mas", "y para ahi", "por ahora"})
        return has_simple_frame and (has_close or self._response_is_action_like(text))

    def _response_handles_strong_block(self, text: str) -> bool:
        normalized_text = self._normalize(text)
        if not normalized_text:
            return False
        lowers_demand = any(
            token in normalized_text
            for token in {"no tienes que", "no hace falta", "no tienes que hacer todo", "solo"}
        )
        return lowers_demand and self._response_is_action_like(text)

    def _response_validates_emotion(self, text: str) -> bool:
        normalized_text = self._normalize(text)
        if not normalized_text:
            return False
        return any(
            token in normalized_text
            for token in {
                "tiene sentido",
                "suena pesado",
                "eso pesa",
                "aqui estoy contigo",
                "entiendo",
                "si te estoy escuchando",
                "si te estoy leyendo",
                "esto si esta siendo pesado",
                "esto sigue siendo pesado",
                "sigo contigo",
            }
        )

    def _build_emotional_priority_response(
        self,
        reason: str,
        domain: str,
        decision_payload: Dict[str, Any],
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        chat_history = chat_history or []
        action = self._default_concrete_action(domain, decision_payload)
        if reason == "frustration":
            return self._pick_non_repeated_variant(
                [
                    f"Si te estoy escuchando. Esto si esta siendo pesado. No te voy a soltar otra lista. Si vamos con una sola cosa pequena, es esta: {action}.",
                    f"Si te estoy leyendo. Esto pesa de verdad. Voy contigo a una sola cosa pequena: {action}.",
                ],
                chat_history,
            )
        if reason == "incomprehension":
            return self._pick_non_repeated_variant(
                [
                    f"Si te estoy escuchando. No te lo dije claro. Te lo digo mas simple: {action}.",
                    f"Perdon, no lo deje claro. Esto ya pesa bastante. Te lo digo simple: {action}.",
                ],
                chat_history,
            )
        if reason == "overwhelm":
            return self._pick_non_repeated_variant(
                [
                    f"Si, esto te esta rebasando mucho. No tienes que poder con todo ahora. Haz solo esto: {action}.",
                    f"Si, esto ya es demasiado para llevarlo entero. Vamos con una sola cosa pequena: {action}.",
                ],
                chat_history,
            )
        return self._pick_non_repeated_variant(
            [
                f"Si te estoy escuchando. Esto sigue siendo pesado. Dejemos una sola cosa pequena y nada mas: {action}.",
                f"Sigo contigo. Esto si pesa y no voy a pasarlo por encima. Vamos solo con esto: {action}.",
            ],
            chat_history,
        )

    def _pick_non_repeated_variant(self, options: List[str], chat_history: List[Dict[str, Any]]) -> str:
        recent_openings = {
            " ".join(self._normalize(text).split()[:4])
            for text in self._recent_assistant_texts(chat_history, limit=2)
            if str(text).strip()
        }
        for option in options:
            opening = " ".join(self._normalize(option).split()[:4])
            if opening not in recent_openings:
                return option
        for option in options:
            if not self._looks_recycled_against_history(option, chat_history):
                return option
        return options[0] if options else ""

    def _build_human_simple_response(
        self,
        normalized_message: str,
        domain: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        chat_history = chat_history or []
        if normalized_message in {"no entiendo", "no te entiendo"}:
            return self._pick_non_repeated_variant(
                [
                    "Esta bien. Te lo digo mas simple.",
                    "Voy a decirlo simple: una sola cosa por vez.",
                ],
                chat_history,
            )
        if normalized_message in {"no se", "no lo se", "no s", "no lo s"}:
            return self._pick_non_repeated_variant(
                [
                    "Esta bien no tenerlo claro ahora. Aqui estoy contigo.",
                    "No pasa nada si ahora no lo ves claro. Aqui sigo contigo.",
                ],
                chat_history,
            )
        if "puedo" in normalized_message and "contigo" in normalized_message and (
            "hablar" in normalized_message or "platicar" in normalized_message
        ):
            return self._pick_non_repeated_variant(
                [
                    "Si, claro. Aqui estoy contigo.",
                    "Si, claro. Puedes platicar conmigo aqui.",
                ],
                chat_history,
            )
        return self._pick_non_repeated_variant(
            [
                "Aqui estoy contigo.",
                "Aqui sigo contigo.",
            ],
            chat_history,
        )

    def _build_direct_answer(
        self,
        normalized_message: str,
        domain: str,
        decision_payload: Dict[str, Any],
        conversation_control: Optional[Dict[str, Any]] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        chat_history = chat_history or []
        conversation_control = conversation_control or {}
        if "quien eres" in normalized_message:
            return self._pick_non_repeated_variant(
                [
                    "Soy NeuroGuIA. Puedes platicar conmigo cuando lo necesites.",
                    "Soy NeuroGuIA. Estoy aqui para acompanar esta conversacion contigo.",
                ],
                chat_history,
            )
        if "para que sirves" in normalized_message:
            return self._pick_non_repeated_variant(
                [
                    "Sirvo para ayudarte a ordenar lo que esta pasando y encontrar un siguiente paso claro cuando haga falta.",
                    "Estoy para ayudarte a entender lo que pasa y dejarlo mas claro o mas llevadero.",
                ],
                chat_history,
            )
        if "como puedo llamarte" in normalized_message or "como te llamo" in normalized_message:
            return self._pick_non_repeated_variant(
                [
                    "Puedes decirme NeuroGuIA, o como te salga mas natural.",
                    "Dime NeuroGuIA si quieres. Como te nazca esta bien.",
                ],
                chat_history,
            )
        if "puedo" in normalized_message and "contigo" in normalized_message and (
            "hablar" in normalized_message or "platicar" in normalized_message
        ):
            return self._pick_non_repeated_variant(
                [
                    "Si, claro. Aqui estoy contigo.",
                    "Si, claro. Puedes platicar conmigo aqui.",
                ],
                chat_history,
            )
        if any(token in normalized_message for token in {"y despues", "y luego", "que sigue", "y ahora", "y ahora que"}):
            signals = conversation_control.get("progression_signals", {}) or {}
            if signals.get("repeated_post_action_followup") or self._is_loop_followup(normalized_message, conversation_control, chat_history):
                return self._build_loop_exit_response(
                    domain=domain,
                    decision_payload=decision_payload,
                    conversation_control=conversation_control,
                    chat_history=chat_history,
                )
            return self._pick_non_repeated_variant(
                [
                    f"Despues va esto: {self._default_concrete_action(domain, decision_payload)}.",
                    f"Lo siguiente es esto: {self._default_concrete_action(domain, decision_payload)}.",
                ],
                chat_history,
            )
        if "hago" in normalized_message or "empiezo" in normalized_message or "comienzo" in normalized_message:
            action = self._default_concrete_action(domain, decision_payload)
            return self._pick_non_repeated_variant(
                [
                    f"Empieza por esto: {action}.",
                    f"Haz una sola cosa: {action}. Nada mas por ahora.",
                ],
                chat_history,
            )
        return self._pick_non_repeated_variant(
            [
                "Si, claro. Aqui estoy contigo.",
                "Si. Aqui sigo contigo.",
            ],
            chat_history,
        )

    def _build_confusion_response(
        self,
        domain: str,
        decision_payload: Dict[str, Any],
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        chat_history = chat_history or []
        action = self._default_concrete_action(domain, decision_payload)
        return self._pick_non_repeated_variant(
            [
                f"Solo haz esto: {action}. Nada mas.",
                f"Haz una sola cosa: {action}. Y para ahi.",
            ],
            chat_history,
        )

    def _build_strong_block_response(
        self,
        domain: str,
        decision_payload: Dict[str, Any],
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        chat_history = chat_history or []
        action = self._default_concrete_action(domain, decision_payload)
        return self._pick_non_repeated_variant(
            [
                f"No tienes que hacer todo. Solo {action}. Nada mas por ahora.",
                f"No hace falta poder con todo ahora. Solo {action}.",
            ],
            chat_history,
        )

    def _build_emotional_response(
        self,
        normalized_message: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        chat_history = chat_history or []
        return self._pick_non_repeated_variant(
            [
                "Tiene sentido que te sientas asi. Aqui estoy contigo.",
                "Suena pesado. No tienes que resolverlo de golpe. Aqui estoy contigo.",
            ],
            chat_history,
        )

    def _build_general_human_response(
        self,
        domain: str,
        decision_payload: Dict[str, Any],
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        chat_history = chat_history or []
        return self._pick_non_repeated_variant(
            [
                "Aqui estoy contigo.",
                f"Haz esto: {self._default_concrete_action(domain, decision_payload)}.",
            ],
            chat_history,
        )

    def _rewrite_abstract_response(
        self,
        turn_intent: str,
        normalized_message: str,
        domain: str,
        decision_payload: Dict[str, Any],
        current_text: str,
        conversation_control: Dict[str, Any],
        chat_history: List[Dict[str, Any]],
    ) -> str:
        if turn_intent != "other":
            return self._build_priority_response(
                turn_intent=turn_intent,
                normalized_message=normalized_message,
                domain=domain,
                decision_payload=decision_payload,
                conversation_control=conversation_control,
                chat_history=chat_history,
            )
        if domain == "ansiedad_cognitiva":
            return "Haz esto: apoya los pies y suelta el aire una vez."
        if domain == "disfuncion_ejecutiva":
            return "Empieza por esto: abre solo el archivo o material que toca."
        if domain == "crisis_activa":
            return "Haz esto: baja un ruido cercano y usa pocas palabras."
        return self._build_general_human_response(domain=domain, decision_payload=decision_payload, chat_history=chat_history)

    def _recent_assistant_texts(self, chat_history: List[Dict[str, Any]], limit: int = 2) -> List[str]:
        texts: List[str] = []
        for turn in reversed(chat_history or []):
            if not isinstance(turn, dict):
                continue
            assistant_text = str(turn.get("assistant") or "").strip()
            if assistant_text:
                texts.append(assistant_text)
            if len(texts) >= limit:
                break
        return texts

    def _is_similar_to_recent_responses(self, curated_text: str, chat_history: List[Dict[str, Any]]) -> bool:
        current_norm = self._normalize(curated_text)
        if not current_norm:
            return False
        current_words = current_norm.split()
        for previous_text in self._recent_assistant_texts(chat_history, limit=2):
            previous_norm = self._normalize(previous_text)
            if not previous_norm:
                continue
            if current_norm == previous_norm:
                return True
            previous_words = previous_norm.split()
            if len(current_words) >= 6 and len(previous_words) >= 6:
                overlap = len(set(current_words).intersection(previous_words))
                base = max(len(set(current_words)), 1)
                if overlap / base >= 0.78:
                    return True
            if " ".join(current_words[:4]) == " ".join(previous_words[:4]):
                return True
        return False

    def _response_is_action_like(self, text: str) -> bool:
        normalized = self._normalize(text)
        return any(
            token in normalized
            for token in {"haz esto", "empieza por", "abre", "apoya", "baja", "quita", "quedate con"}
        )

    def _build_alternative_response(
        self,
        turn_intent: str,
        normalized_message: str,
        domain: str,
        decision_payload: Dict[str, Any],
        current_text: str,
        conversation_control: Dict[str, Any],
        chat_history: List[Dict[str, Any]],
    ) -> str:
        if turn_intent != "other":
            return self._build_priority_response(
                turn_intent=turn_intent,
                normalized_message=normalized_message,
                domain=domain,
                decision_payload=decision_payload,
                conversation_control=conversation_control,
                chat_history=chat_history,
            )

        if self._is_loop_followup(normalized_message, conversation_control, chat_history):
            return self._build_loop_exit_response(domain, decision_payload, conversation_control, chat_history)

        if self._is_clear_question(normalized_message):
            return self._build_direct_answer(normalized_message, domain, decision_payload, conversation_control, chat_history)

        if self._response_is_action_like(current_text):
            if domain == "ansiedad_cognitiva":
                return "Por ahora no hagas mas. Con esto basta."
            if domain == "disfuncion_ejecutiva":
                return "Con lo que ya moviste alcanza por ahora."
            if domain == "crisis_activa":
                return "Por ahora basta. Baja el ruido y para ahi."
            return "Por ahora basta. No hace falta abrir otra vuelta."

        return self._build_general_human_response(domain=domain, decision_payload=decision_payload, chat_history=chat_history)

    def _build_loop_exit_response(
        self,
        domain: str,
        decision_payload: Dict[str, Any],
        conversation_control: Dict[str, Any],
        chat_history: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        chat_history = chat_history or []
        if domain == "ansiedad_cognitiva":
            return self._pick_non_repeated_variant(
                [
                    "Vamos a cerrarlo asi: si eso no vence hoy, sueltalo por ahora. Si si vence hoy, quedate solo con eso.",
                    "Cierro aqui la vuelta: si eso no vence hoy, dejalo quieto. Si si vence hoy, quedate solo con eso.",
                ],
                chat_history,
            )
        if domain == "disfuncion_ejecutiva":
            return self._pick_non_repeated_variant(
                [
                    "Con lo que ya moviste alcanza por ahora. No hace falta abrir otro paso.",
                    "Por ahora basta con eso. No abras otro paso todavia.",
                ],
                chat_history,
            )
        if domain == "crisis_activa":
            return self._pick_non_repeated_variant(
                [
                    "Por ahora basta. Quedate cerca, baja el ruido y no metas otro paso.",
                    "Aqui para. Quedate cerca y no agregues otra indicacion.",
                ],
                chat_history,
            )
        if str(conversation_control.get("turn_family") or "") == "post_action_followup":
            return self._pick_non_repeated_variant(
                [
                    "Por ahora basta. Puedes dejarlo aqui.",
                    "Con esto alcanza por ahora. No hace falta seguir.",
                ],
                chat_history,
            )
        return self._pick_non_repeated_variant(
            [
                f"Cambiemos de forma. Haz esto: {self._default_concrete_action(domain, decision_payload)}.",
                f"Vamos por otra via: {self._default_concrete_action(domain, decision_payload)}.",
            ],
            chat_history,
        )

    def _default_concrete_action(self, domain: str, decision_payload: Dict[str, Any]) -> str:
        response_goal = decision_payload.get("response_goal", {}) or {}
        selected_microaction = str(
            decision_payload.get("selected_microaction")
            or response_goal.get("selected_microaction")
            or ""
        ).strip()
        if selected_microaction:
            return selected_microaction.rstrip(".")
        candidate_actions = [
            str(item).strip()
            for item in response_goal.get("candidate_actions", [])
            if str(item).strip()
        ]
        if candidate_actions:
            return candidate_actions[0].rstrip(".")
        mapping = {
            "ansiedad_cognitiva": "apoya los pies y suelta el aire una vez",
            "disfuncion_ejecutiva": "abre solo el archivo o material que toca",
            "crisis_activa": "baja un ruido cercano y usa pocas palabras",
            "sueno_regulacion": "baja una luz o una pantalla",
        }
        return mapping.get(domain, "haz una sola cosa pequena y visible")

    def _has_template_marker(self, normalized_text: str) -> bool:
        return any(marker in normalized_text for marker in self.TEMPLATE_MARKERS)

    def _last_assistant_text(self, chat_history: List[Dict[str, Any]]) -> str:
        for turn in reversed(chat_history or []):
            if not isinstance(turn, dict):
                continue
            assistant_text = str(turn.get("assistant") or "").strip()
            if assistant_text:
                return assistant_text
        return ""

    def _looks_recycled_against_history(self, curated_text: str, chat_history: List[Dict[str, Any]]) -> bool:
        previous_text = self._last_assistant_text(chat_history)
        if not previous_text:
            return False

        current_norm = self._normalize(curated_text)
        previous_norm = self._normalize(previous_text)
        if not current_norm or not previous_norm:
            return False
        if current_norm == previous_norm:
            return True

        current_words = current_norm.split()
        previous_words = previous_norm.split()
        if len(current_words) >= 8 and len(previous_words) >= 8:
            overlap = len(set(current_words).intersection(previous_words))
            base = max(len(set(current_words)), 1)
            if overlap / base >= 0.82:
                return True

        current_start = " ".join(current_words[:5])
        previous_start = " ".join(previous_words[:5])
        return bool(current_start and current_start == previous_start)

    def _has_repeated_opening(self, curated_text: str, chat_history: List[Dict[str, Any]]) -> bool:
        previous_text = self._last_assistant_text(chat_history)
        if not previous_text:
            return False
        current_start = " ".join(self._normalize(curated_text).split()[:4])
        previous_start = " ".join(self._normalize(previous_text).split()[:4])
        return bool(current_start and current_start == previous_start)


def curate_llm_response(
    llm_result: Optional[Dict[str, Any]] = None,
    fallback_payload: Optional[Dict[str, Any]] = None,
    decision_payload: Optional[Dict[str, Any]] = None,
    stage_result: Optional[Dict[str, Any]] = None,
    state_analysis: Optional[Dict[str, Any]] = None,
    category_analysis: Optional[Dict[str, Any]] = None,
    intent_analysis: Optional[Dict[str, Any]] = None,
    routine_payload: Optional[Dict[str, Any]] = None,
    conversation_control: Optional[Dict[str, Any]] = None,
    conversation_frame: Optional[Dict[str, Any]] = None,
    chat_history: Optional[List[Dict[str, Any]]] = None,
    response_package: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    curator = ResponseCurator()
    return curator.curate(
        llm_result=llm_result,
        fallback_payload=fallback_payload,
        decision_payload=decision_payload,
        stage_result=stage_result,
        state_analysis=state_analysis,
        category_analysis=category_analysis,
        intent_analysis=intent_analysis,
        routine_payload=routine_payload,
        conversation_control=conversation_control,
        conversation_frame=conversation_frame,
        chat_history=chat_history,
        response_package=response_package,
    )
