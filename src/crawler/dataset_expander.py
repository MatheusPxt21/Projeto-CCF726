import json, random, os
import pandas as pd
from datetime import datetime, timedelta

def expand_scenarios(base_events: list, num_scenarios: int = 5000) -> list:
    expanded = []
    salas = [
        {"nome": "Auditório Principal", "cap": 250},
        {"nome": "Auditório Ipioca", "cap": 150},
        {"nome": "Auditório Umbu", "cap": 80},
        {"nome": "Lab Informática 1", "cap": 40},
        {"nome": "Sala Multiuso A", "cap": 60}
    ]
    
    for i in range(num_scenarios):
        base = random.choice(base_events)
        sala_alvo = random.choice(salas)
        hora_base = random.randint(8, 18)
        minuto_base = random.choice([0, 15, 30, 45])
        duracao = random.choice([45, 60, 90, 120])
        
        ini = datetime(2025, 7, 23, hora_base, minuto_base)
        fim = ini + timedelta(minutes=duracao)
        
        participantes = int(np.random.normal(loc=sala_alvo["cap"] * 0.75, scale=25))
        participantes = max(10, participantes)

        expanded.append({
            "evento": base.get("evento", "CSBC"),
            "data": "2025-07-23",
            "horario_inicio": ini.strftime("%H:%M"),
            "horario_fim": fim.strftime("%H:%M"),
            "sala": sala_alvo["nome"],
            "capacidade_sala": sala_alvo["cap"],
            "trilha": base.get("trilha", "Sessão Técnica"),
            "atividade": f"{base.get('atividade', 'Sessão')} [Cenário {i}]",
            "participantes_estimados": participantes
        })
    return expanded