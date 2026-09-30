"""
Script responsável pela coleta de programação de eventos 
(via web scraping e base curada) e expansão sintética dos dados.
"""

import json
import os
import re
import random
import time
from datetime import datetime, timedelta

import numpy as np
import requests
from bs4 import BeautifulSoup

from src.utils.s3_client import S3Storage

RAW_DATA_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constantes de scraping — headers completos para evitar HTTP 403
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.google.com/",
    "Upgrade-Insecure-Requests": "1",
    "sec-ch-ua": '"Chromium";v="124", "Google Chrome";v="124", "Not-A.Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
}

_SBRC_URL  = "https://sbrc.sbc.org.br/2025/pt_br/program-at-glance/"
_SBRC_DATA = "2025-05-20"   # data base de fallback para sessões sem da# ---------------------------------------------------------------------------
# Camada 1 – Web Scraping Real: múltiplos eventos
# ---------------------------------------------------------------------------

_HORARIO_RE    = re.compile(r"(\d{1,2})[h:](\d{2})")
_INTERVALO_RE  = re.compile(r"(\d{1,2})[h:](\d{2})\s*[-–]\s*(\d{1,2})[h:](\d{2})")
_DATA_SBRC     = {
    "19/05/2025": "2025-05-19",
    "20/05/2025": "2025-05-20",
    "21/05/2025": "2025-05-21",
    "22/05/2025": "2025-05-22",
    "23/05/2025": "2025-05-23",
}


def _fmt_hora(h: str, m: str) -> str:
    return f"{int(h):02d}:{m}"


def _horario_fim_estimado(ini: str, duracao_min: int = 90) -> str:
    """Estima horário de fim somando `duracao_min` minutos a `ini` (HH:MM)."""
    try:
        dt = datetime.strptime(ini, "%H:%M") + timedelta(minutes=duracao_min)
        return dt.strftime("%H:%M")
    except Exception:
        return "12:00"


def _get_html(url: str, timeout: int = 20) -> BeautifulSoup | None:
    """Faz GET com HEADERS completos; retorna BeautifulSoup ou None em caso de erro."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        if resp.status_code != 200:
            print(f"  [!] HTTP {resp.status_code} em {url}")
            return None
        resp.encoding = resp.apparent_encoding
        return BeautifulSoup(resp.text, "html.parser")
    except requests.exceptions.ConnectionError:
        print(f"  [!] Falha de conexão em {url}")
    except requests.exceptions.Timeout:
        print(f"  [!] Timeout em {url}")
    except Exception as e:
        print(f"  [!] Erro em {url}: {e}")
    return None


# ── Parser 1: SBRC 2025 /program/ ─────────────────────────────────────────
# Tabela matricial: Horário | Local | 19/05 | 20/05 | 21/05 | 22/05 | 23/05
# Cada célula de data contém o nome da sessão; linhas sem horário são sub-salas.

def _parse_sbrc_program_table(soup: BeautifulSoup) -> list[dict]:
    """
    Parser especializado para a tabela principal da grade do SBRC 2025 em
    https://sbrc.sbc.org.br/2025/pt_br/program/
    Estratégia: itera as linhas mantendo o horário e a sala da linha anterior
    quando uma linha não os traz (rowspan implícito em texto).
    """
    sessoes = []
    table = soup.find("table")
    if not table:
        return sessoes

    rows = table.find_all("tr")
    if not rows:
        return sessoes

    # Descobre quais colunas são datas
    header = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
    # header = ['Horário', 'Local', '19/05/2025', '20/05/2025', '21/05/2025', '22/05/2025', '23/05/2025']
    data_cols = {i: _DATA_SBRC.get(h) for i, h in enumerate(header) if h in _DATA_SBRC}

    horario_atual = ("08:30", "10:00")
    sala_atual = "Auditório Principal SBRC"
    IGNORAR = {"coffee break", "almoço", "lunch", "intervalo", "break", "recess", "jantar", "dinner", ""}

    for row in rows[1:]:
        cells = [td.get_text(separator=" ", strip=True) for td in row.find_all(["td", "th"])]
        if not cells:
            continue

        # Detecta se a linha traz novo horário (coluna 0)
        m_interval = _INTERVALO_RE.search(cells[0])
        if m_interval:
            ini = _fmt_hora(m_interval.group(1), m_interval.group(2))
            fim = _fmt_hora(m_interval.group(3), m_interval.group(4))
            horario_atual = (ini, fim)

        # Detecta sala (coluna 1 quando tem conteúdo não-horário)
        if len(cells) > 1 and cells[1] and not _INTERVALO_RE.search(cells[1]):
            sala_atual = cells[1]

        # Para cada coluna de data, extrai a sessão
        for col_idx, data_iso in data_cols.items():
            if col_idx >= len(cells):
                continue
            atividade = cells[col_idx].strip()
            if not atividade or atividade.lower() in IGNORAR or atividade in ("–", "-", "—", "×", "✕"):
                continue
            # Ignora células muito curtas (apenas símbolo de continuação)
            if len(atividade) <= 2:
                continue

            sessoes.append({
                "evento": "SBRC 2025",
                "data": data_iso,
                "horario_inicio": horario_atual[0],
                "horario_fim": horario_atual[1],
                "sala": sala_atual[:80],
                "capacidade_sala": 150,
                "trilha": "Sessão Técnica SBRC",
                "atividade": atividade[:120],
                "participantes_estimados": 110,
                "fonte": "web_scraping",
            })

    return sessoes


def crawl_sbrc_program() -> list[dict]:
    """Raspa https://sbrc.sbc.org.br/2025/pt_br/program/ (tabela matricial)."""
    url = "https://sbrc.sbc.org.br/2025/pt_br/program/"
    print(f"[*] Raspando grade SBRC 2025 (/program/): {url}")
    soup = _get_html(url)
    if soup is None:
        return []
    sessoes = _parse_sbrc_program_table(soup)
    print(f"[+] Sessões SBRC 2025 /program/: {len(sessoes)}")
    time.sleep(1)
    return sessoes


# ── Parser 2: SBRC 2025 /program-at-glance/ (regex de células) ────────────

def _parse_sbrc_glance_tabelas(soup: BeautifulSoup) -> list[dict]:
    """
    Parser de fallback por regex: itera células de qualquer <table>,
    captura células que contenham horário E texto relevante.
    """
    sessoes = []
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = [c.get_text(separator=" ", strip=True)
                     for c in row.find_all(["td", "th"])]
            if len(cells) < 2:
                continue
            for text in cells:
                m = _HORARIO_RE.search(text)
                if m and len(text) > 10:
                    ini = _fmt_hora(m.group(1), m.group(2))
                    sessoes.append({
                        "evento": "SBRC 2025",
                        "data": "2025-05-20",
                        "horario_inicio": ini,
                        "horario_fim": _horario_fim_estimado(ini),
                        "sala": "Auditório Principal SBRC",
                        "capacidade_sala": 150,
                        "trilha": "Sessão Técnica SBRC",
                        "atividade": text[:90],
                        "participantes_estimados": 100,
                        "fonte": "web_scraping",
                    })
    return sessoes


def _parse_sbrc_glance_divs(soup: BeautifulSoup) -> list[dict]:
    """
    Fallback para divs/spans com regex de horário.
    """
    sessoes = []
    candidatos = soup.find_all(["div", "span", "p", "li"], string=_HORARIO_RE)
    for tag in candidatos:
        texto = tag.get_text(separator=" ", strip=True)
        m = _HORARIO_RE.search(texto)
        if not m or len(texto) <= 10:
            continue
        ini = _fmt_hora(m.group(1), m.group(2))
        titulo = texto
        parent = tag.parent
        if parent:
            t = (parent.find("h2") or parent.find("h3")
                 or parent.find("h4") or parent.find("strong"))
            if t:
                titulo = t.get_text(strip=True)
        sessoes.append({
            "evento": "SBRC 2025",
            "data": "2025-05-20",
            "horario_inicio": ini,
            "horario_fim": _horario_fim_estimado(ini),
            "sala": "Auditório Principal SBRC",
            "capacidade_sala": 150,
            "trilha": "Sessão Técnica SBRC",
            "atividade": titulo[:90],
            "participantes_estimados": 100,
            "fonte": "web_scraping",
        })
    return sessoes


def crawl_sbrc_glance() -> list[dict]:
    """Raspa https://sbrc.sbc.org.br/2025/pt_br/program-at-glance/ (regex)."""
    url = "https://sbrc.sbc.org.br/2025/pt_br/program-at-glance/"
    print(f"[*] Raspando grade SBRC 2025 (/program-at-glance/): {url}")
    soup = _get_html(url)
    if soup is None:
        return []
    sessoes = _parse_sbrc_glance_tabelas(soup)
    if len(sessoes) < 3:
        print("[*] Ativando fallback por divs/spans...")
        sessoes = _parse_sbrc_glance_divs(soup)
    print(f"[+] Sessões reais raspadas da web (SBRC 2025 glance): {len(sessoes)}")
    time.sleep(1)
    return sessoes


# ── Parser 3: IHC 2024 /programacao/ ──────────────────────────────────────
# Tabelas simples com células "11h – 17h30 | Título da sessão | Sala"
# Formato de hora: 11h, 12h30, 13h-14h, 14h30

_IHC_HORA_RE   = re.compile(r"(\d{1,2})h(\d{0,2})")
_IHC_DIA_MAP   = {
    "segunda": "2024-10-07",
    "terça":   "2024-10-08",
    "quarta":  "2024-10-09",
    "quinta":  "2024-10-10",
    "sexta":   "2024-10-11",
}


def _ihc_parse_hora(texto: str) -> tuple[str, str]:
    """Converte 'Xh', 'Xh30', 'Xh – Yh30' em (ini, fim) HH:MM."""
    ms = _IHC_HORA_RE.findall(texto)
    if len(ms) >= 2:
        h1, m1 = ms[0]; h2, m2 = ms[1]
        return _fmt_hora(h1, m1 or "00"), _fmt_hora(h2, m2 or "00")
    elif len(ms) == 1:
        h1, m1 = ms[0]
        ini = _fmt_hora(h1, m1 or "00")
        return ini, _horario_fim_estimado(ini, 90)
    return "09:00", "10:30"


def _parse_ihc_tabelas(soup: BeautifulSoup) -> list[dict]:
    """
    Parser para https://ihc.sbc.org.br/2024/programacao/
    As tabelas têm linhas com: horário pt-BR | título | sala
    Div de dia: 'Segunda – 07/10', 'Terça – 08/10', etc.
    """
    sessoes = []
    data_atual = "2024-10-07"

    # Tenta descobrir a data pelo contexto de cada tabela
    for element in soup.find_all(["h2", "h3", "h4", "strong", "b", "p", "table"]):
        texto_el = element.get_text(strip=True).lower()

        # Atualiza data se o elemento contém nome de dia
        for dia, data_iso in _IHC_DIA_MAP.items():
            if dia in texto_el:
                data_atual = data_iso
                break

        if element.name != "table":
            continue

        # Processa as linhas da tabela
        for row in element.find_all("tr"):
            cells = [td.get_text(separator=" ", strip=True)
                     for td in row.find_all(["td", "th"])]
            if len(cells) < 2:
                continue

            # Coluna 0 deve conter horário
            if not _IHC_HORA_RE.search(cells[0]):
                continue

            ini, fim = _ihc_parse_hora(cells[0])
            atividade = cells[1] if len(cells) > 1 else "Sessão IHC"
            sala = cells[2] if len(cells) > 2 else "Auditório IHC"

            if len(atividade) < 4:
                continue

            sessoes.append({
                "evento": "IHC 2024",
                "data": data_atual,
                "horario_inicio": ini,
                "horario_fim": fim,
                "sala": sala[:80],
                "capacidade_sala": 100,
                "trilha": "IHC",
                "atividade": atividade[:120],
                "participantes_estimados": 70,
                "fonte": "web_scraping",
            })

    return sessoes


def crawl_ihc() -> list[dict]:
    """Raspa https://ihc.sbc.org.br/2024/programacao/"""
    url = "https://ihc.sbc.org.br/2024/programacao/"
    print(f"[*] Raspando grade IHC 2024: {url}")
    soup = _get_html(url)
    if soup is None:
        return []
    sessoes = _parse_ihc_tabelas(soup)
    print(f"[+] Sessões IHC 2024: {len(sessoes)}")
    time.sleep(1)
    return sessoes


# ── Orquestrador ──────────────────────────────────────────────────────────

def crawl_all_events() -> list[dict]:
    """
    Executa todos os scrapers ativos e retorna o conjunto consolidado.
    Cada scraper é tolerante a falhas — erros de rede não interrompem o pipeline.
    """
    print("\n" + "=" * 60)
    print("SCRAPING DE EVENTOS ACADÊMICOS REAIS")
    print("=" * 60)
    todas = []

    scrapers = [
        ("SBRC 2025 /program/",        crawl_sbrc_program),
        ("SBRC 2025 /program-at-glance/", crawl_sbrc_glance),
        ("IHC 2024",                   crawl_ihc),
    ]

    for nome, fn in scrapers:
        try:
            resultado = fn()
            todas.extend(resultado)
        except Exception as e:
            print(f"  [!] Erro no scraper '{nome}': {e}")

    print(f"\n[SCRAPING TOTAL] {len(todas)} sessões obtidas via web scraping")
    return todas


# Mantém alias para compatibilidade com código legado
def crawl_sbrc_schedule() -> list[dict]:
    """Alias de compatibilidade — chama crawl_all_events()."""
    return crawl_all_events()


# ---------------------------------------------------------------------------
# Camada 2 – Base Curada
# ---------------------------------------------------------------------------

def get_base_real_events() -> list:
    """Consolida as atividades reais extraídas das programações de eventos acadêmicos."""
    return [
        # CSBC / WIT - Dia 1
        {"evento": "CSBC - WIT 2025", "data": "2025-07-22", "horario_inicio": "09:00", "horario_fim": "09:45", "sala": "Auditório Ipioca", "capacidade_sala": 150, "trilha": "Abertura", "atividade": "Abertura do WIT 2025", "participantes_estimados": 120},
        {"evento": "CSBC - WIT 2025", "data": "2025-07-22", "horario_inicio": "09:45", "horario_fim": "10:30", "sala": "Auditório Ipioca", "capacidade_sala": 150, "trilha": "Palestra", "atividade": "Palestra de Abertura - RNP (Michelle Wangham)", "participantes_estimados": 140},
        {"evento": "CSBC - WIT 2025", "data": "2025-07-22", "horario_inicio": "10:30", "horario_fim": "12:00", "sala": "Auditório Ipioca", "capacidade_sala": 150, "trilha": "Mesa Redonda", "atividade": "Fórum Meninas Digitais - Encontro dos Projetos Parceiros", "participantes_estimados": 110},
        {"evento": "CSBC - WIT 2025", "data": "2025-07-22", "horario_inicio": "16:00", "horario_fim": "17:00", "sala": "Auditório Ipioca", "capacidade_sala": 150, "trilha": "Sessão Técnica", "atividade": "Sessão Técnica 01 - Seleção dos Melhores Artigos", "participantes_estimados": 95},
        {"evento": "CSBC - WIT 2025", "data": "2025-07-22", "horario_inicio": "17:00", "horario_fim": "18:30", "sala": "Auditório Ipioca", "capacidade_sala": 150, "trilha": "Sessão Técnica", "atividade": "Sessão Técnica 02 - Relatos de Experiência", "participantes_estimados": 80},

        # CSBC / WIT - Dia 2
        {"evento": "CSBC - WIT 2025", "data": "2025-07-23", "horario_inicio": "09:00", "horario_fim": "10:30", "sala": "Auditório Ipioca", "capacidade_sala": 150, "trilha": "Mesa Redonda", "atividade": "Redes Nacionais e Regionais de Extensão com Meninas Digitais", "participantes_estimados": 105},
        {"evento": "CSBC - WIT 2025", "data": "2025-07-23", "horario_inicio": "10:30", "horario_fim": "11:30", "sala": "Auditório Ipioca", "capacidade_sala": 150, "trilha": "Palestra", "atividade": "Tecnologia e Diversidade na ENGIE: Uma Jornada em Evolução", "participantes_estimados": 130},
        {"evento": "CSBC - WIT 2025", "data": "2025-07-23", "horario_inicio": "16:00", "horario_fim": "16:45", "sala": "Auditório Ipioca", "capacidade_sala": 150, "trilha": "Mesa Redonda", "atividade": "Conectando Diferenças: Diversidade e Inclusão na Computação", "participantes_estimados": 115},
        {"evento": "CSBC - WIT 2025", "data": "2025-07-23", "horario_inicio": "16:45", "horario_fim": "18:00", "sala": "Auditório Ipioca", "capacidade_sala": 150, "trilha": "Painel", "atividade": "Painel Embraer: Mulheres, Tecnologia e Carreira", "participantes_estimados": 140},
        {"evento": "CSBC - WIT 2025", "data": "2025-07-23", "horario_inicio": "18:00", "horario_fim": "18:30", "sala": "Auditório Ipioca", "capacidade_sala": 150, "trilha": "Sessão Técnica", "atividade": "Sessão Técnica 03 - Ferramentas", "participantes_estimados": 70},

        # CSBC / WIT - Dia 3
        {"evento": "CSBC - WIT 2025", "data": "2025-07-24", "horario_inicio": "09:00", "horario_fim": "10:00", "sala": "Auditório Ipioca", "capacidade_sala": 150, "trilha": "Sessão Técnica", "atividade": "Sessão Técnica 04 - Análise de Dados Educacionais", "participantes_estimados": 85},
        {"evento": "CSBC - WIT 2025", "data": "2025-07-24", "horario_inicio": "09:00", "horario_fim": "10:00", "sala": "Auditório Umbu", "capacidade_sala": 80, "trilha": "Sessão Técnica", "atividade": "Sessão Técnica 05 - Atração de Talentos Femininos", "participantes_estimados": 60},
        {"evento": "CSBC - WIT 2025", "data": "2025-07-24", "horario_inicio": "10:00", "horario_fim": "10:40", "sala": "Auditório Ipioca", "capacidade_sala": 150, "trilha": "Mesa Redonda", "atividade": "Mesa Redonda: Entrar é só o começo: quem segura a porta?", "participantes_estimados": 135},
        {"evento": "CSBC - WIT 2025", "data": "2025-07-24", "horario_inicio": "10:40", "horario_fim": "12:00", "sala": "Auditório Ipioca", "capacidade_sala": 150, "trilha": "Sessão Técnica", "atividade": "Sessão Técnica 06 - Inclusão e Acessibilidade", "participantes_estimados": 90},
        {"evento": "CSBC - WIT 2025", "data": "2025-07-24", "horario_inicio": "10:40", "horario_fim": "12:00", "sala": "Auditório Umbu", "capacidade_sala": 80, "trilha": "Sessão Técnica", "atividade": "Sessão Técnica 07 - Análise de Dados no Mercado", "participantes_estimados": 75},
        {"evento": "CSBC - WIT 2025", "data": "2025-07-24", "horario_inicio": "16:00", "horario_fim": "16:45", "sala": "Auditório Ipioca", "capacidade_sala": 150, "trilha": "Mesa Redonda", "atividade": "Mesa Redonda: Interseccionalidade em Lideranças", "participantes_estimados": 120},
        {"evento": "CSBC - WIT 2025", "data": "2025-07-24", "horario_inicio": "16:45", "horario_fim": "18:00", "sala": "Auditório Ipioca", "capacidade_sala": 150, "trilha": "Sessão Técnica", "atividade": "Sessão Técnica 08 - Relatos e Intervenções", "participantes_estimados": 80},
        {"evento": "CSBC - WIT 2025", "data": "2025-07-24", "horario_inicio": "16:45", "horario_fim": "18:00", "sala": "Auditório Umbu", "capacidade_sala": 80, "trilha": "Sessão Técnica", "atividade": "Sessão Técnica 09 - Inovação e Sociedade", "participantes_estimados": 70},

        # XII SECOM - UFV Florestal
        {"evento": "XII SECOM UFV", "data": "2023-09-28", "horario_inicio": "10:00", "horario_fim": "11:30", "sala": "Auditório LEN", "capacidade_sala": 180, "trilha": "Palestra", "atividade": "Palestra Abertura: Campanhas de Desinformação no Brasil", "participantes_estimados": 175},
        {"evento": "XII SECOM UFV", "data": "2023-09-28", "horario_inicio": "13:00", "horario_fim": "14:30", "sala": "Auditório LEN", "capacidade_sala": 180, "trilha": "Palestra", "atividade": "Desbravando a Era Digital: Empreendedorismo Tech", "participantes_estimados": 150},
        {"evento": "XII SECOM UFV", "data": "2023-09-28", "horario_inicio": "15:00", "horario_fim": "17:00", "sala": "Auditório LEN", "capacidade_sala": 180, "trilha": "Seminário", "atividade": "III SIPEEC - Seminário de Pesquisa e Extensão", "participantes_estimados": 130},
        {"evento": "XII SECOM UFV", "data": "2023-09-29", "horario_inicio": "07:30", "horario_fim": "11:30", "sala": "Lab Informática 1", "capacidade_sala": 40, "trilha": "Minicurso", "atividade": "Times de Alta Performance em Tecnologia", "participantes_estimados": 38},
        {"evento": "XII SECOM UFV", "data": "2023-09-29", "horario_inicio": "07:30", "horario_fim": "11:30", "sala": "Lab Informática 2", "capacidade_sala": 40, "trilha": "Minicurso", "atividade": "Introdução à Programação Funcional com Haskell", "participantes_estimados": 35},
        {"evento": "XII SECOM UFV", "data": "2023-09-29", "horario_inicio": "12:30", "horario_fim": "16:00", "sala": "Lab Informática 1", "capacidade_sala": 40, "trilha": "Minicurso", "atividade": "Web Front-End: Descomplicando CSS", "participantes_estimados": 40},
        {"evento": "XII SECOM UFV", "data": "2023-09-29", "horario_inicio": "12:30", "horario_fim": "16:00", "sala": "Lab Informática 2", "capacidade_sala": 40, "trilha": "Minicurso", "atividade": "Minicurso Flutter 101", "participantes_estimados": 42},

        # Manna Quantum Festival
        {"evento": "Manna Quantum Festival", "data": "2025-03-14", "horario_inicio": "14:00", "horario_fim": "15:00", "sala": "Auditório Poty Lazzarotto", "capacidade_sala": 130, "trilha": "Oficina", "atividade": "Oficina de Sonhos", "participantes_estimados": 120},
        {"evento": "Manna Quantum Festival", "data": "2025-03-14", "horario_inicio": "15:00", "horario_fim": "16:00", "sala": "Auditório Poty Lazzarotto", "capacidade_sala": 130, "trilha": "Abertura", "atividade": "Abertura Oficial Manna Quantum Festival", "participantes_estimados": 130},
        {"evento": "Manna Quantum Festival", "data": "2025-03-14", "horario_inicio": "16:30", "horario_fim": "17:30", "sala": "Auditório Poty Lazzarotto", "capacidade_sala": 130, "trilha": "Quântica", "atividade": "O Poder da Quântica: do Invisível ao Impossível", "participantes_estimados": 125},
        {"evento": "Manna Quantum Festival", "data": "2025-03-14", "horario_inicio": "17:30", "horario_fim": "19:00", "sala": "Salão de Vidros", "capacidade_sala": 250, "trilha": "Exposição", "atividade": "Manna Experience Interativo", "participantes_estimados": 220},
        {"evento": "Manna Quantum Festival", "data": "2025-03-15", "horario_inicio": "09:30", "horario_fim": "10:30", "sala": "Auditório Poty Lazzarotto", "capacidade_sala": 130, "trilha": "Quântica", "atividade": "Computação Quântica: que História é Essa de Gato?", "participantes_estimados": 115},
        {"evento": "Manna Quantum Festival", "data": "2025-03-15", "horario_inicio": "11:00", "horario_fim": "12:30", "sala": "Auditório Poty Lazzarotto", "capacidade_sala": 130, "trilha": "Quântica", "atividade": "Quantum Computing: The Next Revolution", "participantes_estimados": 128},
        {"evento": "Manna Quantum Festival", "data": "2025-03-15", "horario_inicio": "16:30", "horario_fim": "18:00", "sala": "Auditório Poty Lazzarotto", "capacidade_sala": 130, "trilha": "Inovação", "atividade": "Manna Show: Apresentação de Pitchs", "participantes_estimados": 90},

        # Semana Acadêmica EEL - USP
        {"evento": "Semana Acadêmica EEL-USP", "data": "2024-10-15", "horario_inicio": "10:00", "horario_fim": "12:00", "sala": "Anfiteatro Principal", "capacidade_sala": 200, "trilha": "Indústria", "atividade": "Indústria 4.0 e Inteligência Humana", "participantes_estimados": 165},
        {"evento": "Semana Acadêmica EEL-USP", "data": "2024-10-15", "horario_inicio": "14:00", "horario_fim": "15:00", "sala": "Anfiteatro Principal", "capacidade_sala": 200, "trilha": "Logística", "atividade": "Bem-vindos à P&G: Indústria de Bens e Consumo", "participantes_estimados": 140},
        {"evento": "Semana Acadêmica EEL-USP", "data": "2024-10-15", "horario_inicio": "15:30", "horario_fim": "17:00", "sala": "Anfiteatro Principal", "capacidade_sala": 200, "trilha": "Comunicação", "atividade": "A Arte da Comunicação em Público", "participantes_estimados": 110},
        {"evento": "Semana Acadêmica EEL-USP", "data": "2024-10-16", "horario_inicio": "14:00", "horario_fim": "16:00", "sala": "Sala de Treinamento", "capacidade_sala": 50, "trilha": "Qualidade", "atividade": "Certificação White Belt - Lean Six Sigma", "participantes_estimados": 48},
        {"evento": "Semana Acadêmica EEL-USP", "data": "2024-10-16", "horario_inicio": "19:00", "horario_fim": "20:00", "sala": "Anfiteatro Principal", "capacidade_sala": 200, "trilha": "Liderança", "atividade": "Transparência e Equidade na Carreira (J&J)", "participantes_estimados": 130},

        # XI Mostra Bioeconomia
        {"evento": "XI Mostra Bioeconomia", "data": "2019-10-23", "horario_inicio": "08:00", "horario_fim": "10:00", "sala": "Anfiteatros B", "capacidade_sala": 90, "trilha": "Artigos", "atividade": "Sessão de Apresentação Oral de Extensão", "participantes_estimados": 75},
        {"evento": "XI Mostra Bioeconomia", "data": "2019-10-23", "horario_inicio": "10:30", "horario_fim": "12:30", "sala": "Auditório Principal", "capacidade_sala": 220, "trilha": "Mesa Redonda", "atividade": "Mesa Redonda: Políticas Públicas e Bioeconomia Regional", "participantes_estimados": 190},
        {"evento": "XI Mostra Bioeconomia", "data": "2019-10-24", "horario_inicio": "10:30", "horario_fim": "12:30", "sala": "Auditório Principal", "capacidade_sala": 220, "trilha": "Mesa Redonda", "atividade": "Creditação e Curricularização da Extensão", "participantes_estimados": 160},

        # Fórum CPAs - UFPB
        {"evento": "Fórum CPAs UFPB", "data": "2024-08-13", "horario_inicio": "09:30", "horario_fim": "10:30", "sala": "Auditório CPA", "capacidade_sala": 100, "trilha": "Avaliação", "atividade": "Mesa: Avaliação Institucional e Execução do PDI", "participantes_estimados": 85},
        {"evento": "Fórum CPAs UFPB", "data": "2024-08-13", "horario_inicio": "15:30", "horario_fim": "17:00", "sala": "Auditório CPA", "capacidade_sala": 100, "trilha": "Oficina", "atividade": "Oficina: Proposta Padronizada de Avaliação", "participantes_estimados": 70},
    ]


# ---------------------------------------------------------------------------
# Camada 3 – Expansão Combinatória (Data Augmentation)
# ---------------------------------------------------------------------------

# Salas candidatas com capacidades variadas para simular diferentes contextos
_CAMPUS_ROOMS = [
    {"sala": "Auditório Magna",         "capacidade_sala": 350},
    {"sala": "Auditório Principal",      "capacidade_sala": 250},
    {"sala": "Auditório Ipioca",         "capacidade_sala": 150},
    {"sala": "Auditório Umbu",           "capacidade_sala": 80},
    {"sala": "Auditório Poty Lazzarotto","capacidade_sala": 130},
    {"sala": "Anfiteatro Central",       "capacidade_sala": 180},
    {"sala": "Anfiteatro Norte",         "capacidade_sala": 120},
    {"sala": "Anfiteatro Sul",           "capacidade_sala": 90},
    {"sala": "Lab Informática 1",        "capacidade_sala": 40},
    {"sala": "Lab Informática 2",        "capacidade_sala": 40},
    {"sala": "Lab Informática 3",        "capacidade_sala": 35},
    {"sala": "Sala Seminários A",        "capacidade_sala": 60},
    {"sala": "Sala Seminários B",        "capacidade_sala": 50},
    {"sala": "Sala Seminários C",        "capacidade_sala": 45},
    {"sala": "Espaço Multiuso",          "capacidade_sala": 100},
    {"sala": "Sala de Treinamento",      "capacidade_sala": 50},
    {"sala": "Salão de Vidros",          "capacidade_sala": 300},
    {"sala": "Sala VIP / Reunião",       "capacidade_sala": 25},
]

# Datas do congresso simulado (8 dias)
_CONGRESSO_DATES = [
    "2026-10-10", "2026-10-11", "2026-10-12", "2026-10-13",
    "2026-10-14", "2026-10-15", "2026-10-16", "2026-10-17",
]

# Tipos de imprevistos que motivam a realocação
_SCENARIO_TYPES = [
    "atraso",        # palestrante atrasou, sessão precisa ser encaixada em outro slot
    "choque_sala",   # sala original indisponível / double booking
    "saturacao",     # público muito maior que a capacidade prevista
]


def _gerar_cenario_atraso(base: dict, sala: dict, data: str,
                           idx: int, rng_np) -> dict:
    """
    Simula atraso de palestrante: a sessão começa mais tarde (15–45 min) e
    o público tende a diminuir levemente (pessoas desistem ao esperar).
    """
    hora_ini = random.randint(8, 17)
    atraso   = random.choice([15, 20, 30, 45])
    duracao  = random.choice([45, 60, 90])
    dt_ini   = datetime.strptime(f"{hora_ini:02d}:{random.choice([0,15,30,45]):02d}", "%H:%M")
    dt_ini  += timedelta(minutes=atraso)
    dt_fim   = dt_ini + timedelta(minutes=duracao)

    taxa = rng_np.normal(loc=0.62, scale=0.18)   # público levemente menor
    taxa = max(0.1, min(1.2, taxa))

    return {
        "evento":                base["evento"],
        "data":                  data,
        "horario_inicio":        dt_ini.strftime("%H:%M"),
        "horario_fim":           dt_fim.strftime("%H:%M"),
        "sala":                  sala["sala"],
        "capacidade_sala":       sala["capacidade_sala"],
        "trilha":                base["trilha"],
        "atividade":             f"{base['atividade']} [Atraso #{idx}]",
        "participantes_estimados": int(sala["capacidade_sala"] * taxa),
        "tipo_cenario":          "atraso",
        "fonte":                 base.get("fonte", "curado"),
    }


def _gerar_cenario_choque(base: dict, sala: dict, data: str,
                           idx: int, rng_np) -> dict:
    """
    Simula choque de sala (double booking): a sessão é movida para uma sala
    diferente, possivelmente no mesmo horário mas com capacidade diferente.
    Público mantém-se próximo do original.
    """
    hora_ini  = random.randint(8, 17)
    minuto    = random.choice([0, 15, 30, 45])
    duracao   = random.choice([60, 75, 90, 120])
    dt_ini    = datetime.strptime(f"{hora_ini:02d}:{minuto:02d}", "%H:%M")
    dt_fim    = dt_ini + timedelta(minutes=duracao)

    taxa = rng_np.normal(loc=0.80, scale=0.20)   # público parecido com original
    taxa = max(0.2, min(1.5, taxa))

    return {
        "evento":                base["evento"],
        "data":                  data,
        "horario_inicio":        dt_ini.strftime("%H:%M"),
        "horario_fim":           dt_fim.strftime("%H:%M"),
        "sala":                  sala["sala"],
        "capacidade_sala":       sala["capacidade_sala"],
        "trilha":                base["trilha"],
        "atividade":             f"{base['atividade']} [Choque Sala #{idx}]",
        "participantes_estimados": int(sala["capacidade_sala"] * taxa),
        "tipo_cenario":          "choque_sala",
        "fonte":                 base.get("fonte", "curado"),
    }


def _gerar_cenario_saturacao(base: dict, sala: dict, data: str,
                              idx: int, rng_np) -> dict:
    """
    Simula saturação: público muito maior que a capacidade original, exigindo
    migração para sala maior. Taxa de lotação tende a superar 100%.
    """
    hora_ini  = random.randint(8, 16)
    minuto    = random.choice([0, 15, 30])
    duracao   = random.choice([60, 90, 120])
    dt_ini    = datetime.strptime(f"{hora_ini:02d}:{minuto:02d}", "%H:%M")
    dt_fim    = dt_ini + timedelta(minutes=duracao)

    # Público excede a capacidade (cenário de superlotação)
    taxa = rng_np.normal(loc=1.15, scale=0.20)
    taxa = max(0.8, min(1.6, taxa))

    return {
        "evento":                base["evento"],
        "data":                  data,
        "horario_inicio":        dt_ini.strftime("%H:%M"),
        "horario_fim":           dt_fim.strftime("%H:%M"),
        "sala":                  sala["sala"],
        "capacidade_sala":       sala["capacidade_sala"],
        "trilha":                base["trilha"],
        "atividade":             f"{base['atividade']} [Saturação #{idx}]",
        "participantes_estimados": int(sala["capacidade_sala"] * taxa),
        "tipo_cenario":          "saturacao",
        "fonte":                 base.get("fonte", "curado"),
    }


_CENARIO_FNS = {
    "atraso":      _gerar_cenario_atraso,
    "choque_sala": _gerar_cenario_choque,
    "saturacao":   _gerar_cenario_saturacao,
}


def expand_realocation_scenarios(base_events: list, target_size: int = 50000) -> list:
    """
    Gera cenários sintéticos de realocação.
    Produz `target_size` instâncias com imprevistos do tipo:
      - atraso, choque_sala, saturacao.
    """
    rng = np.random.default_rng(seed=42)
    random.seed(42)

    expanded = []
    tipos = list(_CENARIO_FNS.keys())

    for i in range(target_size):
        base      = random.choice(base_events)
        sala      = random.choice(_CAMPUS_ROOMS)
        data      = random.choice(_CONGRESSO_DATES)
        tipo      = tipos[i % len(tipos)]          # distribui tipos uniformemente
        fn        = _CENARIO_FNS[tipo]
        expanded.append(fn(base, sala, data, i + 1, rng))

    return expanded



# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_pipeline():
    print("=" * 60)
    print("PIPELINE DE COLETA — CCF726")
    print("=" * 60)

    # -- Camada 1: Web Scraping (múltiplos eventos) --
    eventos_web = crawl_all_events()

    # -- Camada 2: Base Curada --
    print("\nConsolidando base curada de eventos reais...")
    eventos_curados = get_base_real_events()
    print(f"Sessões na base curada: {len(eventos_curados)}")

    # -- Fusão das duas camadas --
    base_total = eventos_curados + eventos_web
    print(f"Base total antes da expansão: {len(base_total)} sessões "
          f"({len(eventos_curados)} curadas + {len(eventos_web)} via scraping)")

    # -- Camada 3: Expansão Combinatória --
    print("\nGerando 50.000 instâncias de realocação sintética (Data Augmentation)...")
    dataset_completo = expand_realocation_scenarios(base_total, target_size=50000)

    # -- Persistência Local --
    output_filename = "eventos_academicos_raw.json"
    local_output = os.path.join(RAW_DATA_DIR, output_filename)

    with open(local_output, "w", encoding="utf-8") as f:
        json.dump(dataset_completo, f, ensure_ascii=False, indent=2)
    print(f"\nArquivo local gerado: {local_output}")
    print(f"Total de instâncias no dataset: {len(dataset_completo)}")

    # -- Upload S3 --
    print("\nEnviando para Amazon S3...")
    s3 = S3Storage()
    ok = s3.upload_file(local_output, "raw/eventos_academicos_raw.json")

    # Contagem por fonte de scraping
    por_evento = {}
    for s in eventos_web:
        por_evento[s["evento"]] = por_evento.get(s["evento"], 0) + 1

    print("\n" + "=" * 60)
    print(f"[RESUMO] Sessões curadas         : {len(eventos_curados)}")
    print(f"[RESUMO] Sessões web (total)      : {len(eventos_web)}")
    for ev, cnt in sorted(por_evento.items()):
        print(f"          >> {ev}: {cnt}")
    print(f"[RESUMO] Base pré-expansão        : {len(base_total)}")
    print(f"[RESUMO] Instâncias no dataset    : {len(dataset_completo)}")
    print(f"[RESUMO] Upload S3 (raw)          : {'OK' if ok else 'FALHOU'}")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()