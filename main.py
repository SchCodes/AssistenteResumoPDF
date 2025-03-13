import fitz  # PyMuPDF
import json
import os
import tiktoken  # Para contagem precisa de tokens
from openai import OpenAI

# Carregar a chave da API do arquivo KeyDeepSeek.txt
api_key = None
try:
    with open("KeyDeepSeek.txt", "r") as arquivo:
        api_key = arquivo.read().strip()
except FileNotFoundError:
    print("Arquivo KeyDeepSeek.txt não encontrado. Por favor, crie um arquivo de texto com a chave da API do DeepSeek.")

# Configurações
DEEPSEEK_API_KEY = api_key  # Substitua pela sua chave da API
CAMINHO_PDF = "edital_federal_tecnologia.pdf"  # Caminho do arquivo PDF
BASE_NOME_PDF = os.path.splitext(CAMINHO_PDF)[0]
CAMINHO_JSON_RESUMOS = f"resumos_parciais{CAMINHO_PDF}.json"
CAMINHO_JSON_TEXTO = f"texto_extraido{CAMINHO_PDF}.json"  # Novo arquivo para o texto extraído
MODELO_DEEPSEEK = "deepseek-reasoner" # deepseek-reasoner (R1) ou deepseek-chat (V3) 
LIMITE_TOKENS = 60000  # Margem de segurança abaixo de 65.536 tokens
MAX_NIVEIS = 3  # Níveis máximos de resumo recursivo


# Inicialização do cliente DeepSeek
deepseek = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# Carregar encoder para contagem de tokens
encoder = tiktoken.get_encoding("cl100k_base")

# --- Funções principais ---
def processar_pdf():
    """Função principal para processar o PDF e gerar a resposta."""
    # Verifica se o arquivo de resumos parciais já existe
    if os.path.exists(CAMINHO_JSON_RESUMOS) and os.path.getsize(CAMINHO_JSON_RESUMOS) > 0:
        print("Arquivo de resumos parciais encontrado. Continuando de onde parou...")
        texto_final = carregar_ultimo_resumo()
    else:
        # Cria arquivo JSON de resumos se não existir
        with open(CAMINHO_JSON_RESUMOS, 'w') as f:
            json.dump({}, f)
        
        # Extrai e limpa o texto do PDF
        texto = extrair_texto_pdf()
        texto_limpo = preprocessar_texto(texto)
        
        # Salva o texto extraído em um arquivo JSON
        salvar_texto_extraido(texto_limpo)
        
        print(f"\nTokens inicial total: {contar_tokens(texto_limpo):,}")
        
        # Verifica se o texto precisa ser resumido
        if contar_tokens(texto_limpo) > LIMITE_TOKENS:
            texto_final = resumir_recursivamente(texto_limpo)
        else:
            texto_final = texto_limpo
    
    print(f"\nTokens final: {contar_tokens(texto_final):,}")
    
    # Gera e salva a resposta
    resposta = gerar_resposta(texto_final)
    salvar_resposta(resposta)

def salvar_texto_extraido(texto):
    """Salva o texto extraído do PDF em um arquivo JSON."""
    with open(CAMINHO_JSON_TEXTO, 'w', encoding='utf-8') as f:
        json.dump({"conteudo": texto}, f, ensure_ascii=False, indent=4)
    print(f"Texto extraído salvo em {CAMINHO_JSON_TEXTO}")

def carregar_ultimo_resumo():
    """Carrega o último resumo salvo no arquivo JSON."""
    with open(CAMINHO_JSON_RESUMOS, 'r') as f:
        dados = json.load(f)
        ultimo_nivel = max((int(k.split('_')[1]) for k in dados.keys()), default=0)
        return dados[f'nivel_{ultimo_nivel}']

# --- Funções auxiliares ---
def extrair_texto_pdf():
    """Extrai o texto de um PDF usando PyMuPDF."""
    with fitz.open(CAMINHO_PDF) as doc:
        return " ".join([pagina.get_text() for pagina in doc])

def preprocessar_texto(texto):
    """Remove espaços múltiplos e quebras de linha desnecessárias."""
    return " ".join(texto.replace("\n", " ").split())

def contar_tokens(texto):
    """Calcula a quantidade exata de tokens usando tiktoken."""
    return len(encoder.encode(texto))

def dividir_em_partes(texto, max_tokens=30000):
    """Divide o texto em partes menores, respeitando o limite de tokens."""
    partes = []
    sentencas = texto.split('. ')
    parte_atual = []
    tokens_atual = 0

    for sentenca in sentencas:
        tokens_sentenca = contar_tokens(sentenca)
        if tokens_atual + tokens_sentenca > max_tokens:
            partes.append(". ".join(parte_atual) + ".")
            parte_atual = []
            tokens_atual = 0
        parte_atual.append(sentenca)
        tokens_atual += tokens_sentenca

    if parte_atual:
        partes.append(". ".join(parte_atual) + ".")
    return partes

def resumir_recursivamente(texto, nivel=1):
    """Resume o texto recursivamente até que o número de tokens seja aceitável."""
    if nivel > MAX_NIVEIS:
        print("Limite máximo de níveis atingido!")
        return texto
        
    # Verifica se já existe resumo deste nível
    resumo_salvo = gerenciar_resumos(nivel)
    if resumo_salvo:
        print(f"Carregando resumo nível {nivel} do cache")
        return resumo_salvo

    print(f"\n--- Processando nível {nivel} ---")
    print(f"Tokens iniciais: {contar_tokens(texto):,}")
    
    partes = dividir_em_partes(texto)
    resumos = []
    
    for i, parte in enumerate(partes, 1):
        print(f"Resumindo parte {i}/{len(partes)} ({contar_tokens(parte):,} tokens)")
        resumo = gerar_resumo(parte)
        resumos.append(resumo)
        print(f"Tokens após resumo: {contar_tokens(resumo):,}\n")
    
    texto_combinado = " ".join(resumos)
    
    # Controle de redução progressiva
    reducao = (contar_tokens(texto) - contar_tokens(texto_combinado)) / contar_tokens(texto) * 100
    print(f"Redução total nível {nivel}: {reducao:.2f}%")
    
    if reducao < 15 and nivel > 1:
        print("Redução insuficiente, interrompendo processo")
        return texto_combinado
    
    # Salva progresso atual
    gerenciar_resumos(nivel, texto_combinado, 'escrita')
    
    if contar_tokens(texto_combinado) > LIMITE_TOKENS:
        return resumir_recursivamente(texto_combinado, nivel + 1)
        
    return texto_combinado

def gerenciar_resumos(nivel, conteudo=None, operacao='leitura'):
    """Gerencia o armazenamento e recuperação de resumos parciais."""
    try:
        if operacao == 'escrita':
            with open(CAMINHO_JSON_RESUMOS, 'r+') as f:
                dados = json.load(f) if os.path.getsize(CAMINHO_JSON_RESUMOS) > 0 else {}
                dados[f'nivel_{nivel}'] = conteudo
                f.seek(0)
                json.dump(dados, f, indent=2)
            return True
            
        with open(CAMINHO_JSON_RESUMOS, 'r') as f:
            dados = json.load(f)
            return dados.get(f'nivel_{nivel}', None)
            
    except Exception as e:
        print(f"Erro no gerenciamento de resumos: {str(e)}")
        return None

def gerar_resumo(texto):
    """Gera um resumo do texto usando a API DeepSeek."""
    resposta = deepseek.chat.completions.create(
        model=MODELO_DEEPSEEK,
        messages=[
            {"role": "system", "content": "Você é um especialista em resumos técnicos."},
            {"role": "user", "content": f"Resuma este texto, o mínimo de palavras deve ser 1000\nTexto: {texto}"}
        ],
        temperature=0,
        max_tokens=8000
    )
    return resposta.choices[0].message.content

def gerar_resposta(texto):
    """Gera uma resposta estruturada com base no texto resumido."""
    resposta = deepseek.chat.completions.create(
        model=MODELO_DEEPSEEK,
        messages=[
            {"role": "system", "content": "Gerar resposta estruturada em markdown com:"},
            {"role": "user", "content": f"Com base neste documento {texto}, crie um resumo detalhado com:\n# Objetivo\n# Requisitos\n# Prazos\n# Etapas\n# Entre outras informações que sejam relevantes\nMínimo de palavras deve ser 4000\nTexto: {texto}\nCorrija a formatação do Markdown removendo os espaços extras após os títulos e substituindo os asteriscos por hashtags para os títulos."}
        ],
        temperature=0.1,
        max_tokens=8000
    )
    return resposta.choices[0].message.content

def salvar_resposta(resposta):
    """Salva a resposta em um arquivo Markdown."""
    # Remove a extensão .pdf do nome do arquivo e adiciona .md
    nome_base = os.path.splitext(CAMINHO_PDF)[0] # Remove a extensão .pdf do nome do arquivo
    caminho_markdown = f"{nome_base}.md"

    with open(caminho_markdown, "w", encoding="utf-8") as md:
        md.write(f"# Resumo do Edital: {os.path.basename(nome_base)}\n\n")
        md.write(resposta)
    print(f"Resposta salva em {caminho_markdown}")

# Execução principal
if __name__ == "__main__":
    try:
        processar_pdf()
    except Exception as e:
        print(f"Erro crítico: {str(e)}")