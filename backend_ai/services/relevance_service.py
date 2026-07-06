import os
import re
import json
import logging
from typing import List, Dict, Any
from google.genai import types
from backend_ai.core.config_loader import AGENTS_CONFIG
from backend_ai.core.api_utils import get_gemini_client

logger = logging.getLogger("services.relevance")

class RelevanceScorer:
    def __init__(self):
        self.gemini_client = get_gemini_client()
        self.gemini_model = "gemini-2.5-flash"
        self.fallback_models = ["gemini-2.5-pro", "gemini-1.5-pro", "gemini-1.5-flash"]
        
        # Override config if present
        if AGENTS_CONFIG:
            ma_config = AGENTS_CONFIG.get("media_analyst", {})
            self.gemini_model = ma_config.get("primary_model", self.gemini_model)
            if "fallback_models" in ma_config:
                self.fallback_models = ma_config["fallback_models"]

    def _clean_profanity(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        profanities = {
            r"\bfuck(ing|er|ed|s)?\b": "[expletive]",
            r"\bshit(s|ted|ting|head)?\b": "[expletive]",
            r"\bass(hole)?s?\b": "[expletive]",
            r"\bbitch(es)?\b": "[expletive]",
            r"\bcrap\b": "[expletive]",
            r"\bdamn\b": "[expletive]",
        }
        cleaned = text
        for pattern, replacement in profanities.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        return cleaned

    def _clean_nested_profanity(self, data: Any) -> Any:
        if isinstance(data, str):
            return self._clean_profanity(data)
        elif isinstance(data, dict):
            return {k: self._clean_nested_profanity(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_nested_profanity(x) for x in data]
        return data

    def score_segments(self, user_prompt: str, media_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Calculates prompt relevance scores for segments in media_analyses.
        Modifies media_analyses in-place and returns it.
        """
        user_prompt = self._clean_profanity(user_prompt)
        media_analyses = self._clean_nested_profanity(media_analyses)

        if not media_analyses:
            return media_analyses

        flat_segments = []
        for analysis in media_analyses:
            filename = analysis.get("file_metadata", {}).get("filename")
            if not filename:
                continue
                
            # Highlights
            for idx, h in enumerate(analysis.get("interesting_segments", [])):
                start = h.get("start", 0.0)
                end = h.get("end", 0.0)
                desc = h.get("description", "")
                # face_present is injected by ClipScoringAgent.score_visual_data()
                face_visible = h.get("face_present", False)
                flat_segments.append({
                    "id": f"{filename}:highlight:{idx}:{start}:{end}",
                    "filename": filename,
                    "type": "highlight",
                    "index": idx,
                    "description": desc,
                    "face_visible": face_visible,
                })

            # Chronological segments
            for idx, s in enumerate(analysis.get("all_segments", [])):
                start = s.get("start", 0.0)
                end = s.get("end", 0.0)
                desc = s.get("description", "")
                face_visible = s.get("face_present", False)
                flat_segments.append({
                    "id": f"{filename}:segment:{idx}:{start}:{end}",
                    "filename": filename,
                    "type": "segment",
                    "index": idx,
                    "description": desc,
                    "face_visible": face_visible,
                })

        if not flat_segments:
            logger.info("RelevanceScorer: No segments found to score.")
            return media_analyses

        # Build prompt
        prompt = f"""You are an expert video editor assistant.
Analyze the user's creative brief and score the relevance of each available video segment.

User Creative Brief: "{user_prompt}"

IMPORTANT SCORING RULES:
1. PEOPLE/CREATOR RULE: If the creative brief contains first-person words ('me', 'us',
   'we', 'I', 'myself', 'ourselves') or people references ('person', 'people', 'creator',
   'athlete', 'speaker', 'friend', 'team', or any name), then segments where
   face_visible=True are HIGHLY RELEVANT by definition — score them >= 0.85, because
   they show the creator that the brief is about. Do NOT score face segments low just
   because their description mentions a landscape or activity.
2. DIRECT MATCH: Segments whose description matches the brief's stated action, location,
   or theme score >= 0.70.
3. INDIRECT: Segments loosely related to the mood or setting score 0.40 - 0.69.
4. IRRELEVANT: Segments completely unrelated to the brief score <= 0.30.

Available Video Segments:
"""
        for seg in flat_segments:
            face_tag = "face_visible: True" if seg.get("face_visible") else "face_visible: False"
            prompt += f"- ID: {seg['id']} | {face_tag} | Description: {seg['description']}\n"

        prompt += """
For each segment, output a relevance score between 0.0 (completely irrelevant) and 1.0 (highly relevant to the user prompt's actions, characters, or mood).
Provide the output as a raw JSON object matching this structure:
{
  "scores": [
    {
      "id": "segment_id_here",
      "relevance_score": float
    }
  ]
}
Do not include any other text or markdown formatting. Only return the raw JSON object.
"""

        scores_data = None

        all_models = [self.gemini_model]
        for m in self.fallback_models:
            if m not in all_models:
                all_models.append(m)

        for model_id in all_models:
            try:
                logger.info(f"RelevanceScorer: Attempting relevance scoring via Gemini using {model_id}...")
                response = self.gemini_client.models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json"
                    )
                )
                scores_data = json.loads(response.text)
                logger.info("RelevanceScorer: Successfully retrieved scores from Gemini.")
                break
            except Exception as e:
                logger.warning(f"RelevanceScorer: Gemini scoring failed with {model_id}: {e}")

        # If we failed to get scores, assign a default score of 0.5 to all
        if not scores_data:
            logger.warning("RelevanceScorer: All Gemini models failed. Falling back to default relevance scores (0.5).")
            scores_data = {"scores": []}

        # Apply scores
        return self._apply_scores(media_analyses, flat_segments, scores_data)

    def _apply_scores(self, media_analyses: List[Dict[str, Any]], flat_segments: List[Dict[str, Any]], data: Dict[str, Any]) -> List[Dict[str, Any]]:
        scores_map = {}
        for item in data.get("scores", []):
            seg_id = item.get("id")
            score = item.get("relevance_score", 0.5)
            try:
                score = max(0.0, min(1.0, float(score)))
            except (ValueError, TypeError):
                score = 0.5
            scores_map[seg_id] = score

        # Modify media_analyses in-place
        for seg in flat_segments:
            score = scores_map.get(seg["id"], 0.5)
            for analysis in media_analyses:
                filename = analysis.get("file_metadata", {}).get("filename")
                if filename == seg["filename"]:
                    if seg["type"] == "highlight":
                        interesting = analysis.get("interesting_segments", [])
                        if seg["index"] < len(interesting):
                            interesting[seg["index"]]["relevance_score"] = score
                    elif seg["type"] == "segment":
                        all_segs = analysis.get("all_segments", [])
                        if seg["index"] < len(all_segs):
                            all_segs[seg["index"]]["relevance_score"] = score

        return media_analyses
