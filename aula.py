import math
import os
import re
import unicodedata
from collections import Counter, defaultdict

import streamlit as st

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

st.set_page_config(page_title="RI com PDFs", page_icon="📄", layout="wide")

STOPWORDS = {
    "a", "as", "o", "os", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
    "e", "ou", "em", "no", "na", "nos", "nas", "por", "para", "com", "sem", "ao", "aos",
    "que", "se", "sua", "seu", "suas", "seus", "como", "mais", "menos", "durante", "pode",
    "podem", "feito", "uma", "garantir", "é", "ser", "foi"
}


def remover_acentos(texto):
    texto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def tokenizar(texto):
    return re.findall(r"[a-zA-ZÀ-ÿ0-9]+", texto, re.UNICODE)


def normalizar(token):
    return remover_acentos(token.lower())


def stemmer(token):
    sufixos = [
        "mente", "imentos", "imento", "amentos", "amento", "acoes", "acao",
        "idades", "idade", "ando", "endo", "indo", "aram", "eram", "iram",
        "ados", "adas", "idos", "idas", "es", "s"
    ]
    for sufixo in sorted(sufixos, key=len, reverse=True):
        if len(token) > len(sufixo) + 2 and token.endswith(sufixo):
            return token[:-len(sufixo)]
    return token


def processar(texto, remover_stopwords, aplicar_stemming):
    tokens = tokenizar(texto)
    normalizados = [normalizar(token) for token in tokens]
    stopwords_norm = {normalizar(palavra) for palavra in STOPWORDS}
    sem_stopwords = [
        token for token in normalizados
        if not remover_stopwords or token not in stopwords_norm
    ]
    finais = [stemmer(token) if aplicar_stemming else token for token in sem_stopwords]
    return tokens, normalizados, sem_stopwords, finais


def extrair_pdf(caminho):
    if PdfReader is None:
        raise RuntimeError("Instale a dependência necessária com: pip install pypdf")
    leitor = PdfReader(caminho)
    paginas = []
    for numero, pagina in enumerate(leitor.pages, 1):
        texto = pagina.extract_text() or ""
        paginas.append((numero, texto))
    return paginas


def carregar_documentos(pasta):
    documentos = {}
    erros = []
    if not os.path.isdir(pasta):
        return documentos, [f"Diretório não encontrado: {pasta}"]

    arquivos = sorted(nome for nome in os.listdir(pasta) if nome.lower().endswith(".pdf"))
    for nome in arquivos:
        caminho = os.path.join(pasta, nome)
        try:
            paginas = extrair_pdf(caminho)
            texto = "\n".join(texto for _, texto in paginas).strip()
            if texto:
                documentos[nome] = {"texto": texto, "paginas": paginas, "caminho": caminho}
            else:
                erros.append(f"{nome}: não foi possível extrair texto; pode ser um PDF escaneado.")
        except Exception as erro:
            erros.append(f"{nome}: {erro}")
    return documentos, erros


def construir_indice(documentos_tokens):
    indice = defaultdict(list)
    for doc_id, tokens in documentos_tokens.items():
        for token in sorted(set(tokens)):
            indice[token].append(doc_id)
    return dict(sorted(indice.items()))


def calcular_ranking(documentos_tokens, query_tokens):
    n = len(documentos_tokens)
    df = Counter()
    for tokens in documentos_tokens.values():
        for termo in set(tokens):
            df[termo] += 1

    termos = sorted(set(query_tokens))
    idf = {termo: math.log(n / df[termo]) if df[termo] else 0.0 for termo in termos}
    ranking = []

    for doc_id, tokens in documentos_tokens.items():
        frequencias = Counter(tokens)
        tamanho = len(tokens)
        score = 0.0
        detalhes = []
        for termo in termos:
            tf = frequencias[termo] / tamanho if tamanho else 0.0
            tfidf = tf * idf[termo]
            score += tfidf
            detalhes.append({
                "Termo": termo,
                "TF": round(tf, 6),
                "DF": df[termo],
                "IDF": round(idf[termo], 6),
                "TF-IDF": round(tfidf, 6),
            })
        ranking.append({
            "Documento": doc_id,
            "TF-IDF acumulado": round(score, 6),
            "Detalhes": detalhes,
        })

    ranking.sort(key=lambda item: item["TF-IDF acumulado"], reverse=True)
    return ranking, idf


st.title("📄 Recuperação de Informação com PDFs")
st.caption("Os PDFs colocados no diretório configurado servem como base dos documentos pesquisáveis.")

with st.sidebar:
    st.header("Configuração")
    pasta_pdf = st.text_input("Diretório dos PDFs", value="docs")
    usar_stopwords = st.checkbox("Remover stopwords", value=True)
    usar_stemming = st.checkbox("Aplicar stemming", value=True)
    atualizar = st.button("Recarregar PDFs")

if PdfReader is None:
    st.error("A biblioteca pypdf não está instalada.")
    st.code("pip install pypdf")
    st.stop()

if atualizar:
    st.cache_data.clear()

@st.cache_data(show_spinner="Lendo PDFs...")
def carregar_com_cache(pasta):
    return carregar_documentos(pasta)

arquivos, erros = carregar_com_cache(pasta_pdf)

if erros:
    for erro in erros:
        st.warning(erro)

if not arquivos:
    st.info("Nenhum PDF foi encontrado. Crie uma pasta 'docs' ao lado deste arquivo e coloque os PDFs nela.")
    st.code("seu_projeto/\n├── app_consolidador_ri_pdf.py\n└── docs/\n    ├── documento1.pdf\n    └── documento2.pdf")
    st.stop()

st.success(f"{len(arquivos)} PDF(s) carregado(s) do diretório `{pasta_pdf}`.")
query = st.text_input("Digite uma query", value="soja irrigação lagartas")

processados = {
    nome: processar(dados["texto"], usar_stopwords, usar_stemming)
    for nome, dados in arquivos.items()
}
documentos_tokens = {nome: dados[3] for nome, dados in processados.items()}

aba_docs, aba_pipeline, aba_indice, aba_busca = st.tabs([
    "Documentos", "Pré-processamento", "Índice invertido", "Busca TF-IDF"
])

with aba_docs:
    st.subheader("PDFs carregados")
    for nome, dados in arquivos.items():
        with st.expander(nome):
            st.write(f"Páginas: {len(dados['paginas'])}")
            st.text(dados["texto"][:4000])
            if len(dados["texto"]) > 4000:
                st.caption("Prévia limitada aos primeiros 4.000 caracteres.")

with aba_pipeline:
    doc_selecionado = st.selectbox("Documento para inspecionar", list(processados))
    nomes_etapas = ["Tokenização", "Normalização", "Sem stopwords", "Stemming"]
    colunas = st.columns(4)
    for coluna, nome, valores in zip(colunas, nomes_etapas, processados[doc_selecionado]):
        with coluna:
            st.markdown(f"**{nome}**")
            st.write(valores[:500])
    vocabulario = sorted(set(token for tokens in documentos_tokens.values() for token in tokens))
    st.subheader(f"Vocabulário ({len(vocabulario)} termos)")
    st.write(vocabulario)

with aba_indice:
    indice = construir_indice(documentos_tokens)
    st.subheader("Termo → lista de PDFs")
    st.json(indice)
    st.subheader("Consulta rápida no índice")
    termo = st.text_input("Termo", value="soja")
    termo_processado = processar(termo, usar_stopwords, usar_stemming)[3]
    if termo_processado:
        st.write(indice.get(termo_processado[0], []))

with aba_busca:
    query_tokens = processar(query, usar_stopwords, usar_stemming)[3]
    st.write("Tokens da query:", query_tokens)
    ranking, idf = calcular_ranking(documentos_tokens, query_tokens)

    st.subheader("Ranking")
    for posicao, item in enumerate(ranking, 1):
        vencedor = posicao == 1 and item["TF-IDF acumulado"] > 0
        prefixo = "🏆 " if vencedor else ""
        st.write(f"{prefixo}{posicao}. **{item['Documento']}** — {item['TF-IDF acumulado']:.6f}")

    if ranking and ranking[0]["TF-IDF acumulado"] > 0:
        st.success(f"Documento vencedor: {ranking[0]['Documento']}")
    else:
        st.info("Nenhum termo da query foi encontrado nos PDFs.")

    st.subheader("Detalhamento TF, DF, IDF e TF-IDF")
    for item in ranking:
        with st.expander(f"{item['Documento']} — score {item['TF-IDF acumulado']:.6f}"):
            for detalhe in item["Detalhes"]:
                st.write(
                    f"{detalhe['Termo']}: TF={detalhe['TF']:.6f} | DF={detalhe['DF']} | "
                    f"IDF={detalhe['IDF']:.6f} | TF-IDF={detalhe['TF-IDF']:.6f}"
                )

st.divider()
st.caption("TF = frequência do termo / total de tokens; IDF = ln(N / DF); TF-IDF = TF × IDF.")