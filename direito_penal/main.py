import yaml
import spacy

try:
    nlp = spacy.load("pt_core_news_lg")
except:
    print("Erro ao carregar spaCy. Verifique seu venv.")

with open("fluxo.yaml", "r", encoding="utf-8") as f:
    fluxo = yaml.safe_load(f)


def entender_confirmacao(texto):
    afirmacoes = [
        "sim",
        "foi",
        "houve",
        "com certeza",
        "positivo",
        "quebrou",
        "acho que sim",
        "s",
    ]
    negacoes = ["nao", "não", "nada", "negativo", "sem violência", "jamais", "n"]
    texto = texto.lower().strip()
    if any(p in texto for p in afirmacoes):
        return "sim"
    if any(p in texto for p in negacoes):
        return "nao"
    return None


def executar_atendimento():
    print("\n" + "=" * 50)
    print("      NOVO CHAT: ASSISTENTE JURÍDICO")
    print("=" * 50)

    relato = (
        input("Relate o ocorrido (ou digite 'sair' para encerrar): ").lower().strip()
    )

    if relato == "sair":
        return False

    doc = nlp(relato)

    # Busca por violencia
    lemas_violencia = [
        "bater",
        "agredir",
        "empurrar",
        "chutar",
        "ameaçar",
        "armar",
        "apontar",
        "socil",
        "roubar",
        "assaltar",
    ]
    violencia_detectada_ia = any(
        t.lemma_ in lemas_violencia or t.text in ["arma", "faca", "assalto", "agressão"]
        for t in doc
    )

    print("\n[SISTEMA]: Para garantir a correta tipificação, preciso saber:")
    print("-> Houve violência ou grave ameaça contra alguma pessoa?")

    sugestao = (
        "(IA: Detectei indícios de violência no relato) "
        if violencia_detectada_ia
        else ""
    )

    while True:
        confirmacao_v = input(f"Você {sugestao}(sim/nao): ").lower().strip()
        decisao_v = entender_confirmacao(confirmacao_v)

        if decisao_v:
            houve_violencia = decisao_v == "sim"
            break
        else:
            print("Bot: Por favor, responda 'sim' ou 'não' para prosseguirmos.")

    contexto = fluxo["crimes_patrimoniais"]

    # Direcionamento do fluxo
    if houve_violencia:
        caminho = contexto["decisoes"]["com_violencia"]
    else:
        caminho = contexto["decisoes"]["sem_violencia"]

    print(f"\nBot: {caminho['proxima_pergunta']}")

    while True:
        resp_detalhe = input("Você: ").lower().strip()
        decisao_final = entender_confirmacao(resp_detalhe)

        if decisao_final:
            resultado = caminho["opcoes"].get(decisao_final)
            print("\n" + "-" * 30)
            print(f"[CONCLUSÃO JURÍDICA]: {resultado}")
            print("-" * 30)
            break
        else:
            print("Bot: Não entendi. Responda 'sim' ou 'não'.")

    return True


if __name__ == "__main__":
    rodando = True
    while rodando:
        rodando = executar_atendimento()

    print("\nEncerrando sistema... EngStrategy agradece o contato.")
