"""Prueba cualitativa con 20 frases nuevas (10 positivas, 10 negativas). No es una métrica estadística.

IMPORTANTE: la especificación pide que estas frases las escriba la autora. Las siguientes son un
BORRADOR redactado por el asistente para poder ejecutar la prueba; la autora debe reemplazarlas
por las suyas antes de entregar el informe.
"""
from __future__ import annotations

from .predictor import ModeloNoDisponible, Predictor

# (frase, esperado) 1 = Positivo, 0 = Negativo
FRASES: list[tuple[str, int]] = [
    ("Me llegó antes de lo previsto y funciona perfecto, muy contento", 1),
    ("Sin ninguna queja, justo lo que buscaba", 1),
    ("No le tenía fe, pero me sorprendió gratamente", 1),
    ("Excelente calidad por el precio, lo recomiendo totalmente", 1),
    ("Mis hijos lo adoran y no se ha estropeado en meses", 1),
    ("No puedo estar más satisfecha con la compra", 1),
    ("Compraré otro igual sin dudarlo, vale cada centavo", 1),
    ("Es tal cual la descripción, muy buen material", 1),
    ("Nunca me había funcionado tan bien un producto así", 1),
    ("Cumple con todo lo prometido y el envío fue rapidísimo", 1),
    ("Jamás volvería a pedirlo, una pérdida de dinero", 0),
    ("Llegó con un defecto de fábrica y nadie me responde", 0),
    ("Me estafaron, el producto no se parece en nada a la foto", 0),
    ("Bonito pero se rompió a la primera semana", 0),
    ("No funciona, lo tuve que devolver", 0),
    ("Muy mala calidad, el material es horrible", 0),
    ("Esperaba mucho más, una decepción total", 0),
    ("No lo recomiendo para nada, huele raro y se despintó", 0),
    ("Tardó un mes en llegar y además venía incompleto", 0),
    ("Ni siquiera enciende, no sirve", 0),
]


def main() -> float | None:
    try:
        pred = Predictor()
    except ModeloNoDisponible as e:
        print("Omitido:", e)
        return None
    aciertos = 0
    print(f"{'Frase':<64}{'Esp.':>6}{'Pred.':>10}{'P(pos)':>8}  ok")
    for frase, esp in FRASES:
        r = pred.predecir(frase)
        pr = 1 if r["etiqueta"] == "Positivo" else 0
        ok = pr == esp
        aciertos += ok
        print(f"{frase[:62]:<64}{'Pos' if esp else 'Neg':>6}{r['etiqueta']:>10}{r['prob_positivo']:>8.2f}  {'✓' if ok else '✗'}")
    pct = aciertos / len(FRASES) * 100
    print(f"\nAciertos: {aciertos}/{len(FRASES)} = {pct:.0f}%  (prueba cualitativa, no estadística)")
    return pct


if __name__ == "__main__":
    main()
