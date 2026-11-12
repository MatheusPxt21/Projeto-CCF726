"""
Gerador de incidentes de realocação sobre grades REAIS de eventos.

Unidade de análise: um *incidente* = (grade real de um dia de evento, atividade
perturbada, tipo de imprevisto) + K opções candidatas (sala livre, horário livre).

Cada candidato recebe:
  * atributos OBSERVÁVEIS (o que um agente saberia no momento da decisão), e
  * um `score_viabilidade` REALIZADO, produzido por um simulador que depende de
    variáveis LATENTES não observadas pelo modelo (comparecimento real, afinidade
    de público entre trilhas, sensibilidade ao deslocamento, atratividade do
    horário) + ruído. Isso evita que o alvo seja uma função determinística das
    features (o que levaria a R² ~ 1 artificial).
"""
from __future__ import annotations

import json
import zlib
from collections import defaultdict

import numpy as np

# Salas extras disponíveis no local do evento (oferecem variedade de capacidade)
CAMPUS_ROOMS = [
    ("Auditório Magna", 350), ("Auditório Principal", 250), ("Anfiteatro Central", 180),
    ("Auditório Ipioca", 150), ("Auditório Poty Lazzarotto", 130), ("Anfiteatro Norte", 120),
    ("Espaço Multiuso", 100), ("Anfiteatro Sul", 90), ("Auditório Umbu", 80),
    ("Sala Seminários A", 60), ("Sala Seminários B", 50), ("Sala Seminários C", 45),
    ("Lab Informática 1", 40), ("Lab Informática 3", 35), ("Sala VIP / Reunião", 25),
]
TIPOS = {"atraso": 0, "choque_sala": 1, "saturacao": 2}
DIA_INI, DIA_FIM, PASSO = 8 * 60, 20 * 60, 30
TRILHAS_GENERICAS = {"Sessão Técnica SBRC", "IHC"}


def _min(s: str) -> int:
    h, m = map(int, s.strip()[:5].split(":"))
    return h * 60 + m


def _trilha(e: dict) -> str:
    # Nas grades raspadas a trilha real (workshop/simpósio) está no nome da sala
    if e["trilha"] in TRILHAS_GENERICAS:
        return e["sala"].split("\n")[0][:25].strip()
    return e["trilha"]


def carregar_grades(path: str) -> dict:
    """Agrupa as sessões reais por (evento, data) -> lista de sessões normalizadas."""
    with open(path, encoding="utf-8") as f:
        base = json.load(f)
    grades = defaultdict(list)
    for e in base:
        try:
            ini, fim = _min(e["horario_inicio"]), _min(e["horario_fim"])
        except Exception:
            continue
        if fim <= ini:
            continue
        grades[(e["evento"], e["data"])].append({
            "sala": e["sala"].split("\n")[0][:40].strip(),
            "cap": int(e.get("capacidade_sala") or 100),
            "trilha": _trilha(e),
            "ini": ini, "fim": fim,
            "publico": int(e.get("participantes_estimados") or 50),
            "atividade": e["atividade"], "fonte": e.get("fonte", "curado"),
        })
    return dict(grades)


def _afinidade(t1: str, t2: str, seed: int) -> float:
    """Sobreposição LATENTE de público entre duas trilhas distintas (fixa por par)."""
    a, b = sorted((t1, t2))
    h = zlib.crc32(f"{a}|{b}|{seed}".encode())
    return 0.05 + 0.40 * (h % 10_000) / 10_000


def _atratividade(t: int, pref_tarde: float) -> float:
    """Fator LATENTE de comparecimento por horário (almoço e fim de dia esvaziam)."""
    f = 1.0
    if 12 * 60 <= t < 13 * 60 + 30:
        f -= 0.22
    if t >= 17 * 60 + 30:
        f -= 0.25 * (1 - pref_tarde)
    if t < 9 * 60:
        f -= 0.12
    return f


def gerar_incidentes(grades: dict, n_incidentes: int, k: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    chaves = sorted(grades)
    linhas: list[dict] = []
    inc_id = 0
    tentativas = 0
    while inc_id < n_incidentes and tentativas < n_incidentes * 20:
        tentativas += 1
        evento, data = chaves[rng.integers(len(chaves))]
        grade = grades[(evento, data)]
        alvo_idx = int(rng.integers(len(grade)))
        alvo = grade[alvo_idx]
        outras = [s for i, s in enumerate(grade) if i != alvo_idx]
        dur = alvo["fim"] - alvo["ini"]
        if dur > 240:
            continue

        tipo = ("atraso", "choque_sala", "saturacao")[inc_id % 3]
        demanda = alvo["publico"]
        if tipo == "saturacao":
            demanda = int(demanda * rng.uniform(1.2, 1.7))
        elif tipo == "atraso":
            demanda = int(demanda * rng.uniform(0.8, 1.0))

        # Local: salas reais do evento + subconjunto aleatório de salas do campus
        salas_evento = {}
        for (ev, _), sess in grades.items():
            if ev == evento:
                for s in sess:
                    salas_evento[s["sala"]] = s["cap"]
        extras = rng.choice(len(CAMPUS_ROOMS), size=int(rng.integers(4, 9)), replace=False)
        for j in extras:
            nome, cap = CAMPUS_ROOMS[j]
            salas_evento.setdefault(nome, cap)
        n_salas = len(salas_evento)

        # Enumera todas as opções (sala, início) livres na grade real
        opcoes = []
        for sala, cap in salas_evento.items():
            ocupado = [(s["ini"], s["fim"]) for s in outras if s["sala"] == sala]
            for t in range(DIA_INI, DIA_FIM - dur + 1, PASSO):
                if tipo == "atraso" and t < alvo["ini"]:
                    continue  # atraso: não dá para antecipar
                if sala == alvo["sala"] and t == alvo["ini"]:
                    continue  # o slot original é justamente o que falhou
                if tipo == "choque_sala" and sala == alvo["sala"]:
                    continue  # sala original indisponível
                if any(max(t, a) < min(t + dur, b) for a, b in ocupado):
                    continue
                opcoes.append((sala, cap, t))
        if len(opcoes) < k:
            continue
        escolhidas = rng.choice(len(opcoes), size=k, replace=False)

        # Variáveis latentes do incidente (NÃO observadas pelo modelo)
        ruido_publico = rng.lognormal(0, 0.18)
        sens_desloc = rng.uniform(0.03, 0.30)        # perda de público por hora de deslocamento
        afin_mesma = rng.uniform(0.55, 1.0)          # canibalização entre sessões da mesma trilha
        pref_tarde = rng.uniform(0, 1)
        seed_afin = seed

        for j in escolhidas:
            sala, cap, t = opcoes[j]
            conc = [s for s in outras if max(t, s["ini"]) < min(t + dur, s["fim"])]
            mesma = [s for s in conc if s["trilha"] == alvo["trilha"]]
            desloc = abs(t - alvo["ini"])
            pico = int((10 * 60 <= t < 12 * 60) or (14 * 60 <= t < 16 * 60))
            densidade = len(conc) / n_salas

            # ----- Simulador de impacto (latente) -----
            retencao = np.exp(-sens_desloc * desloc / 60)
            canib = 0.0
            for s in conc:
                frac = (min(t + dur, s["fim"]) - max(t, s["ini"])) / dur
                af = afin_mesma if s["trilha"] == alvo["trilha"] else _afinidade(alvo["trilha"], s["trilha"], seed_afin)
                canib += frac * af * min(1.0, s["publico"] / max(demanda, 1))
            publico_real = demanda * ruido_publico * _atratividade(t, pref_tarde) * retencao
            ratio = publico_real / cap
            pen_lotacao = 55 * max(0.0, ratio - 1.0) ** 0.8
            pen_ociosa = 12 * max(0.0, 0.35 - ratio) / 0.35
            pen_trilha = 22 * canib
            pen_perda = 30 * (1 - retencao)
            pen_fluxo = 10 * densidade * rng.lognormal(0, 0.3) + 4 * pico
            score = 100 - (pen_lotacao + pen_ociosa + pen_trilha + pen_perda + pen_fluxo)
            score += rng.normal(0, 4)

            linhas.append({
                "incidente_id": inc_id,
                "evento": evento, "data": data, "fonte": alvo["fonte"],
                "atividade": alvo["atividade"], "trilha": alvo["trilha"],
                "tipo_cenario": tipo, "tipo_cenario_cod": TIPOS[tipo],
                "sala_original": alvo["sala"],
                "horario_original": f"{alvo['ini'] // 60:02d}:{alvo['ini'] % 60:02d}",
                "sala": sala,
                "horario_inicio": f"{t // 60:02d}:{t % 60:02d}",
                "horario_fim": f"{(t + dur) // 60:02d}:{(t + dur) % 60:02d}",
                "duracao_minutos": dur,
                "capacidade_sala": cap,
                "participantes_estimados": demanda,
                "fator_ocupacao_sala": round(demanda / cap, 3),
                "indice_sobreposicao_trilha": len(mesma),
                "sessoes_concorrentes": len(conc),
                "densidade_bloco": round(densidade, 3),
                "horario_pico": pico,
                "hora_inicio_min": t,
                "deslocamento_minutos": desloc,
                "mesma_sala": int(sala == alvo["sala"]),
                "score_viabilidade": float(np.clip(round(score, 2), 0, 100)),
            })
        inc_id += 1
    return linhas
