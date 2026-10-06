"""Generate the small synthetic documents used by the document-shape evaluation.

Run once; the PDFs are committed so the evaluation is reproducible without
reportlab.
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

OUT = Path(__file__).with_name("documents")

OUT.mkdir(parents=True, exist_ok=True)


def lines(c, entries, start_y=780, size=12, leading=18):
    c.setFont("Helvetica", size)
    y = start_y

    for entry in entries:
        c.drawString(25 * mm, y, entry)
        y -= leading

    return y


# --------------------------------------------------------------------------
# A short sheet: two pages, the shape of an invoice or a form.
# --------------------------------------------------------------------------
c = canvas.Canvas(str(OUT / "sheet.pdf"), pagesize=A4)
c.setTitle("Site Safety Induction Record")
c.setAuthor("Northgate Facilities")

lines(
    c,
    [
        "Site Safety Induction Record",
        "",
        "This record covers the induction required before working on site.",
        "The induction must be renewed every 12 months.",
        "Visitors must sign in at reception and be escorted at all times.",
        "Personal protective equipment is required beyond the yellow line.",
    ],
)
c.showPage()
lines(
    c,
    [
        "Appendix A: contacts",
        "",
        "The site manager is D. Okonkwo.",
        "The first aid room is on the ground floor beside the loading bay.",
        "Report near misses to the site manager within 24 hours.",
    ],
)
c.showPage()
c.save()

# --------------------------------------------------------------------------
# A table-heavy document: reading a value out of a table.
# --------------------------------------------------------------------------
c = canvas.Canvas(str(OUT / "table.pdf"), pagesize=A4)
c.setTitle("Monthly Sensor Readings")
lines(
    c,
    ["Monthly Sensor Readings", "", "Date        Sensor    Temperature    Humidity"],
    size=11,
    leading=16,
)

data = [
    ("2024-03-01", "N-01", "11.2", "48"),
    ("2024-03-02", "N-01", "12.8", "46"),
    ("2024-03-03", "N-01", "14.5", "45"),
    ("2024-03-04", "N-01", "13.1", "47"),
    ("2024-03-05", "N-02", "15.9", "44"),
]
lines(
    c,
    [f"{d}     {s}       {t}            {h}" for d, s, t, h in data],
    start_y=735,
    size=11,
    leading=16,
)
c.showPage()
lines(
    c,
    [
        "Notes",
        "",
        "Sensor N-02 was installed on 5 March 2024.",
        "All temperatures are in degrees Celsius.",
    ],
)
c.showPage()
c.save()

# --------------------------------------------------------------------------
# A non-English document.
# --------------------------------------------------------------------------
c = canvas.Canvas(str(OUT / "german.pdf"), pagesize=A4)
c.setTitle("Wartungsbericht der Anlage")
c.setAuthor("Technik Nord")
lines(
    c,
    [
        "Wartungsbericht der Anlage",
        "",
        "Der Bericht beschreibt die Wartung der Heizungsanlage.",
        "Die Wartung erfolgt alle sechs Monate durch einen Techniker.",
        "Der letzte Termin war am 14. April.",
        "Die Pruefung des Drucks ist jaehrlich vorgeschrieben.",
    ],
)
c.showPage()
c.save()

print("written to", OUT)
for path in sorted(OUT.glob("*.pdf")):
    print(f"  {path.name}: {path.stat().st_size} bytes")
