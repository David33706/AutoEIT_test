"""Write transcription results to Excel."""

import openpyxl


def write_to_excel(all_results, template_path, output_path, sheet_names):
    """
    Write transcription results into the Excel template.

    Args:
        all_results: Dict of {participant_id: results_list}
        template_path: Path to original Excel template
        output_path: Path for output Excel file
        sheet_names: Dict mapping participant_id to sheet name
    """
    wb = openpyxl.load_workbook(template_path)

    for pid, results in all_results.items():
        ws = wb[sheet_names[pid]]
        for r in results:
            row = r["item"] + 1  # Row 2 = item 1
            ws.cell(row=row, column=3, value=r["transcription"])

    wb.save(output_path)
    print(f"Saved to {output_path}")