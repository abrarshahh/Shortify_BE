import librosa
import numpy as np
import json
import os
from typing import List, Dict, Any

class RhythmEngineer:
    def __init__(self):
        pass

    def analyze_music(self, file_path: str) -> Dict[str, Any]:
        """
        Analyzes a music file to detect beats, tempo, and high-energy segments.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        print(f"Analyzing music: {file_path}")
        
        # Load audio file
        y, sr = librosa.load(file_path)

        # 1. Beat Tracking
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)

        # 2. Onset Strength (for energy/drops)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        times = librosa.times_like(onset_env, sr=sr)
        
        # Find peaks in onset envelope (potential "drops" or hits)
        peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
        peak_times = times[peaks]

        # 3. RMS Energy (root-mean-square energy)
        rms = librosa.feature.rms(y=y)[0]
        rms_times = librosa.times_like(rms, sr=sr)
        
        # Find high energy segments
        mean_rms = np.mean(rms)
        high_energy_indices = np.where(rms > mean_rms * 1.5)[0]
        high_energy_times = rms_times[high_energy_indices]

        # Group high energy times into segments
        energy_segments = self._group_timestamps(high_energy_times.tolist())

        # Ensure tempo is a float (librosa 0.10+ returns an array)
        if isinstance(tempo, np.ndarray):
            tempo = tempo[0] if tempo.size > 0 else 0

        # 4. Sentiment Analysis
        sentiment = self._estimate_sentiment(y, sr, float(tempo))

        result = {
            "tempo": float(tempo),
            "beat_count": len(beat_times),
            "beat_times": beat_times.tolist(),
            "peak_times": peak_times.tolist(),
            "energy_segments": energy_segments,
            "duration": float(librosa.get_duration(y=y, sr=sr)),
            "sentiment": sentiment,
            "segment_sentiments": self._get_segment_sentiments(y, sr, float(tempo))
        }

        return result

    def _group_timestamps(self, timestamps: List[float], threshold: float = 2.0) -> List[Dict[str, float]]:
        """
        Groups individual high-energy timestamps into segments.
        """
        if not timestamps:
            return []

        segments = []
        if not timestamps:
            return segments

        start = timestamps[0]
        last = timestamps[0]

        for i in range(1, len(timestamps)):
            if timestamps[i] - last > threshold:
                segments.append({"start": round(start, 2), "end": round(last, 2)})
                start = timestamps[i]
            last = timestamps[i]
        
        segments.append({"start": round(start, 2), "end": round(last, 2)})
        return [s for s in segments if s["end"] - s["start"] > 0.5] # filter out tiny blips

    def _estimate_sentiment(self, y: np.ndarray, sr: int, tempo: float) -> Dict[str, Any]:
        """
        Estimates the mood/sentiment based on tempo, brightness, and energy.
        """
        # 1. Arousal (Energy/Tempo) - How intense is the sound?
        rms = librosa.feature.rms(y=y)[0]
        avg_rms = np.mean(rms)
        # Heuristic: Higher energy and higher tempo = higher arousal
        arousal = (avg_rms * 5 + (tempo / 200)) / 2
        arousal = np.clip(arousal, 0, 1)

        # 2. Valence (Brightness/Tempo) - How positive is the sound?
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        avg_centroid = np.mean(centroid)
        # Heuristic: Brighter sounds (higher centroid) and moderate-high tempo = higher valence
        valence = (avg_centroid / 4000 + (tempo / 200)) / 2
        valence = np.clip(valence, 0, 1)

        # Mapping to Labels
        if valence > 0.6:
            label = "Happy/Energetic" if arousal > 0.5 else "Calm/Peaceful"
        elif valence < 0.4:
            label = "Tense/Aggressive" if arousal > 0.5 else "Sad/Melancholic"
        else:
            label = "Neutral/Ambient"

        # Score is the primary sentiment strength (0 to 1)
        score = valence if valence > 0.5 else (1 - valence)

        return {
            "label": label,
            "score": round(float(score), 3),
            "valence": round(float(valence), 3),
            "arousal": round(float(arousal), 3)
        }

    def _get_segment_sentiments(self, y: np.ndarray, sr: int, tempo: float, window_size: int = 5) -> List[Dict[str, Any]]:
        """
        Breaks audio into windows and calculates sentiment for each.
        """
        duration = librosa.get_duration(y=y, sr=sr)
        segments = []
        
        for start_t in range(0, int(duration), window_size):
            end_t = min(start_t + window_size, duration)
            start_sample = int(start_t * sr)
            end_sample = int(end_t * sr)
            y_seg = y[start_sample:end_sample]
            
            if len(y_seg) > sr * 0.5: # At least 0.5s of audio
                sentiment = self._estimate_sentiment(y_seg, sr, tempo)
                segments.append({
                    "start": round(float(start_t), 2),
                    "end": round(float(end_t), 2),
                    "sentiment": sentiment
                })
        
        return segments

if __name__ == "__main__":
    # Test script
    import sys
    if len(sys.argv) > 1:
        engineer = RhythmEngineer()
        analysis = engineer.analyze_music(sys.argv[1])
        print(json.dumps(analysis, indent=2))
