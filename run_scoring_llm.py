"""
AutoEIT Test II: LLM-based EIT Scoring
Uses GPT-4o-mini to apply the Ortega (2000) rubric.
"""

import os
import sys
import json
import openpyxl
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from src.score_llm import score_with_llm
from src.score_rules import score_eit, content_overlap, levenshtein_sim

INPUT_PATH = "data/AutoEIT_Sample_Transcriptions_for_Scoring.xlsx"
OUTPUT_DIR = "output"

PARTICIPANTS = ["38001-1A", "38002-2A", "38004-2A", "38006-2A"]

TARGETS = [
    "Quiero cortarme el pelo",
    "El libro está en la mesa",
    "El carro lo tiene Pedro",
    "El se ducha cada mañana",
    "Qué dice usted que va a hacer hoy",
    "Dudo que sepa manejar muy bien",
    "Las calles de esta ciudad son muy anchas",
    "Puede que llueva mañana todo el día",
    "Las casas son muy bonitas pero caras",
    "Me gustan las películas que acaban bien",
    "El chico con el que yo salgo es español",
    "Después de cenar me fui a dormir tranquilo",
    "Quiero una casa en la que vivan mis animales",
    "A nosotros nos fascinan las fiestas grandiosas",
    "Ella sólo bebe cerveza y no come nada",
    "Me gustaría que el precio de las casas bajara",
    "Cruza a la derecha y después sigue todo recto",
    "Ella ha terminado de pintar su apartamento",
    "Me gustaría que empezara a hacer más calor pronto",
    "El niño al que se le murió el gato está triste",
    "Una amiga mía cuida a los niños de mi vecino",
    "El gato que era negro fue perseguido por el perro",
    "Antes de poder salir él tiene que limpiar su cuarto",
    "La cantidad de personas que fuman ha disminuido",
    "Después de llegar a casa del trabajo tomé la cena",
    "El ladrón al que atrapó la policía era famoso",
    "Le pedí a un amigo que me ayudara con la tarea",
    "El examen no fue tan difícil como me habían dicho",
    "Serías tan amable de darme el libro que está en la mesa",
    "Hay mucha gente que no toma nada para el desayuno",
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize OpenAI client
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter your OpenAI API key: ").strip()
    client = OpenAI(api_key=api_key)

    wb = openpyxl.load_workbook(INPUT_PATH)
    all_results = {}

    for sheet_name in PARTICIPANTS:
        ws = wb[sheet_name]
        results = []

        print(f"\n{'='*80}")
        print(f"PARTICIPANT {sheet_name}")
        print(f"{'='*80}")
        print(f"{'#':>2} | {'Rule':>4} | {'LLM':>3} | {'Match':>5} | Transcription")
        print("-" * 80)

        for i in range(30):
            row = i + 2
            transcription = ws.cell(row=row, column=3).value or ""
            target = TARGETS[i]

            # Rule-based score
            rule_score = score_eit(target, transcription)

            # LLM score
            llm_result = score_with_llm(target, transcription, client)
            llm_score = llm_result["score"]

            match = "✓" if rule_score == llm_score else f"Δ{abs(rule_score - llm_score)}"

            results.append({
                "item": i + 1,
                "target": target,
                "transcription": transcription,
                "rule_score": rule_score,
                "llm_score": llm_score,
                "llm_reason": llm_result["reason"],
            })

            print(f"{i+1:>2} |   {rule_score}  |  {llm_score}  | {match:>5} | {transcription[:55]}")

        all_results[sheet_name] = results

        # Summary
        rule_avg = sum(r["rule_score"] for r in results) / 30
        llm_avg = sum(r["llm_score"] for r in results if r["llm_score"] >= 0) / 30
        agree = sum(1 for r in results if r["rule_score"] == r["llm_score"])
        close = sum(1 for r in results if abs(r["rule_score"] - r["llm_score"]) <= 1)

        print(f"\nRule avg: {rule_avg:.2f} | LLM avg: {llm_avg:.2f}")
        print(f"Exact agreement: {agree}/30 ({agree/30*100:.0f}%)")
        print(f"Within 1 point:  {close}/30 ({close/30*100:.0f}%)")

    # Save comparison
    with open(os.path.join(OUTPUT_DIR, "scores_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nComparison saved to output/scores_comparison.json")


if __name__ == "__main__":
    main()