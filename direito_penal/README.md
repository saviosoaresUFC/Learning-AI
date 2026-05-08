# ⚖️ Assistente de Triagem Jurídica - Direito Penal

Este módulo é um sistema especialista baseado em **Regras e Intenções** desenvolvido para realizar o enquadramento típico preliminar de crimes contra o patrimônio. O assistente utiliza Processamento de Linguagem Natural (NLP) para analisar relatos e guiar o usuário através de um protocolo de segurança jurídica.

## 🧠 Lógica de Funcionamento

O sistema opera sob o **Protocolo de Segurança EngStrategy**, que prioriza a análise de bens jurídicos indisponíveis (vida e integridade física) antes de analisar o patrimônio:

1.  **Captura de Relato:** O usuário descreve o ocorrido em linguagem natural.
2.  **Análise Semântica (spaCy):** A IA identifica lemas (raízes de palavras) que indicam violência ou grave ameaça (ex: bater, ameaçar, empurrar).
3.  **Filtragem Obrigatória:** Independentemente da análise da IA, o sistema obriga a confirmação de violência física/moral para evitar erros de tipificação entre Furto (Art. 155) e Roubo (Art. 157).
4.  **Afunilamento via YAML:** O fluxo de perguntas e as respostas legais são carregados de um arquivo externo, facilitando a manutenção da legislação sem alterar o código-fonte.

## 🛠️ Tecnologias Utilizadas

- **Python 3.14+**
- **spaCy (Modelo `pt_core_news_lg`):** Para análise semântica e lematização de alta precisão.
- **PyYAML:** Para gestão da base de conhecimento jurídica.

## 🚀 Como Executar

1.  Certifique-se de que o ambiente virtual (`venv`) está ativo.

    ```bash
    python -m venv venv
    # No Windows
    venv\Scripts\activate
    # No Linux/Mac
    source venv/bin/activate
    ```

2.  Instale as dependências específicas:

    ```bash
    pip install -r requirements.txt
    python -m spacy download pt_core_news_lg
    ```

3.  Inicie o assistente:

    ```bash
    python main.py
    ```

### ⚠️ Resolução de Problemas (Bloqueio de DLL)

Em alguns casos, as políticas de segurança do Windows podem bloquear o download ou a execução das DLLs do spaCy/Pydantic. Se você encontrar um erro de "Política de Controle de Aplicativo", siga estes passos:

1.  Abra o menu **Iniciar** e pesquise por **Segurança do Windows**.
2.  Clique em **Controle de aplicativos e navegador**.
3.  Vá em **Configurações do controle de aplicativos**.
4.  Altere para a opção **"Desativado"**.
5.  Tente o comando de download novamente.

    ```bash
    python -m spacy download pt_core_news_lg
    ```

## 📂 Estrutura de Arquivos

- `main.py`: Motor de execução e lógica de diálogo.
- `fluxo.yaml`: Base de conhecimento contendo artigos, penas e árvore de decisão.
- `README.md`: Documentação do módulo.

## ⚖️ Avisos Legais

Este sistema é uma ferramenta de estudo e triagem preliminar. Não substitui a consulta a um advogado ou autoridade policial. As penas e artigos referem-se ao **Código Penal Brasileiro**.

---

**Desenvolvido por Sávio de Carvalho Soares**
