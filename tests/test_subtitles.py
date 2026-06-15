import json
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend_ai.agents.subtitle_agent import SubtitleAgent

RENDERED_VIDEO = "path/to/video.mp4"
EDL_PATH       = "edl_output.json"


def test_subtitles():
    print("--- Starting Phase 6: Subtitle & Safety Agent Test ---")

    agent = SubtitleAgent(model_size="base", device="cpu")

    # ------------------------------------------------------------------ #
    # 1. SAFE-ZONE CHECK (no model needed, runs immediately)              #
    # ------------------------------------------------------------------ #
    if os.path.exists(EDL_PATH):
        print("\n[1] Running Safe-Zone Check on EDL text overlays...")
        with open(EDL_PATH, "r") as f:
            edl = json.load(f)

        report = agent.check_safe_zones(edl)
        print(f"    Verdict : {report['verdict']}")
        print(f"    Summary : {report['summary']}")

        if report["flagged_items"]:
            print("    Flagged overlays:")
            for item in report["flagged_items"]:
                print(f"      - '{item['text_overlay']}' -> {item['flags']}")
        else:
            print("    All text overlays are within the platform safe zone.")

        # Save full report
        with open("safe_zone_report.json", "w") as f:
            json.dump(report, f, indent=4)
        print("    Full report saved to: safe_zone_report.json")
    else:
        print(f"    EDL not found at {EDL_PATH}, skipping safe-zone check.")

    # ------------------------------------------------------------------ #
    # 2. TRANSCRIPTION (requires Whisper + rendered video)               #
    # ------------------------------------------------------------------ #
    if not os.path.exists(RENDERED_VIDEO):
        print(f"\n[2] Rendered video not found at '{RENDERED_VIDEO}'.")
        print("    Run tests/test_editor.py first, then re-run this test.")
        return

    print(f"\n[2] Transcribing rendered video: {RENDERED_VIDEO}")
    transcription = agent.transcribe(RENDERED_VIDEO)

    print(f"    Detected language : {transcription['language']}")
    print(f"    Full text         : {transcription['full_text'][:200]}")
    print(f"    Caption segments  : {len(transcription['captions'])}")

    # Save transcription
    with open("transcription_output.json", "w") as f:
        json.dump(transcription, f, indent=4)
    print("    Transcription saved to: transcription_output.json")

    # ------------------------------------------------------------------ #
    # 3. BURN SUBTITLES onto the video                                   #
    # ------------------------------------------------------------------ #
    if transcription["captions"]:
        print("\n[3] Burning subtitles onto video...")
        output_path = "data/exports/shortify_reel_subtitled.mp4"
        agent.burn_subtitles(
            video_path=RENDERED_VIDEO,
            captions=transcription["captions"],
            output_path=output_path,
        )
        print(f"    Subtitled video saved to: {output_path}")
    else:
        print("\n[3] No captions generated (possibly silent video). Skipping burn step.")

    print("\n--- Phase 6 Test Complete ---")


if __name__ == "__main__":
    test_subtitles()
