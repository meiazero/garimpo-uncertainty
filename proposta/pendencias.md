# Pendências da proposta

Conteúdo que saiu do texto científico. Não vai para o orientador como parte da proposta.

## Decisões para levar ao orientador

1. Aprovar o tema.
2. Confirmar o que vence em 30/11/2026: entrega do texto à banca ou defesa. O cronograma assume entrega do texto.
3. Aceitar o formato exploratório, com uma semente por modelo e sem testes confirmatórios.
4. Aceitar a delimitação a sítios já localizados, sem busca em área desconhecida.

## Decisões em aberto

- Corte de recortes (máximo de 3 por grupo na calibração e no teste; 300 industriais no treino). Não aplicado.
- LOGAR: não citado, porque falta documento oficial.
- COMPASS-J ponderado: célula fora da Tabela 1. O paper relata que COMPASS-L com pesos falhou sob mudança no H&E e que só as variantes de COMPASS-J
  mantiveram a cobertura nas duas mudanças (`papers/cheung2026compass.pdf`, seção de covariate shift, Figura 3 e Tabela 5).

## Declarações a preencher pelo autor

- Financiamento e aviso institucional do Censipam.
- Uso de IA, conforme as regras do programa.
- Disponibilidade de dados e código.

## Detalhes técnicos do desenho

Saíram do texto para deixá-lo mais direto. Voltam na dissertação.

- **Modelos:** rede simples sem conexões de salto e U-Net. Fora: DeepLabV3+, SegFormer, Prithvi-EO-2.0, SAM2.
- **Entrada:** bandas B2, B3, B4, B8, B11 e B12; células de 512 x 512 pixels (2.621,44 ha).
- **Intervalos:** pixels independentes (controle), conformal simples, relativo, local, regressão quantílica conformal, COMPASS-L e
  COMPASS-J; 15 combinações por modelo, 30 no total. Fora: E2E-CQR, FCP, conjuntos de pixels, controle de risco.
- **COMPASS:** calibra a magnitude de uma perturbação nos logits (L) ou em camada interna (J). Em J, verificar monotonicidade.
- **Quantil:** peso igual por grupo (CDF pooling, Dunn 2022). A combinação com o COMPASS não tem garantia formal estabelecida.
- **Cenários:** externo, interno (deixa um grupo de fora) e externo ponderado por razão de densidades.
- **Partição (recortes artesanais/grupos):** treino 56/21 mais 947 industriais; validação 18/5; calibração 69/25; teste 89/24.
- **Comparação:** todos os métodos nas mesmas condições, sem comparação principal definida. Regra da Q2: cobertura com IC de 95% incluindo 90%; entre esses, menor
  largura. Q1 usa correlação de Spearman por modelo. Falta definir a tolerância do F1 de fronteira.
- **Métricas:** PICP, MPIW, interval score, MAE, viés, erro relativo, Dice, F1 de fronteira; bootstrap sobre grupos, 1.000
  reamostragens.
- **Delimitação:** mede-se a cicatriz visível na imagem. Ilegalidade depende de informação externa (situação jurídica) e redes
  de ilícitos não são modeladas.
- **Precisão:** com 24 grupos, poder 0,771 para queda até 0,70 e 0,344 até 0,80. Distinguir 0,85 de 0,90 exige cerca de 299
  unidades. Com 25 grupos de calibração, cobertura realizada média 0,923, desvio-padrão 0,051.

## Plano de contingência

- Critério após o primeiro treino: Dice de validação de pelo menos 0,60 na U-Net. Abaixo disso, um dia para revisar dados e
  treino. Se persistir, segue com o melhor modelo.
- Cortes, nesta ordem: COMPASS-J só na U-Net; retirar a rede simples; 300 recortes industriais. A data de 14/11 não muda.

## Operacional

- Acesso e fila da `gpuq` não conferidos (sessão `tsh` expirada).
- Disco: cerca de 59 GB sem compressão para todas as células; cerca de 27 GB com 300 recortes industriais.
- Cenas industriais ainda não conferidas no Planetary Computer; as 232 artesanais foram.
- Durações do cronograma são estimativas não testadas.

## Situação das fontes

`evidencias.csv` liga as afirmações às fontes. Todas ainda precisam de conferência humana.

| Situação | Fontes |
|---|---|
| PDF local, trechos conferidos | Cheung, Maus, Couttenier, Cui, Lehmann, Gonzalez-Calabuig, Lu, Silva, Tang, Dunn, Barber |
| PDF local, só identificação conferida | Demais artigos com identificador arXiv |
| Só página, resumo ou README | Jasansky 2024, Okumura, Amazon Mining Watch, LIGEM.Redes, SMP, Wundram, Cosarinsky, Viti, Mossina |
| Só metadados | Gneiting e Raftery 2007, Angelopoulos 2023 (publicado), Lei 2018 (publicado) |
| Texto completo não obtido | Jasansky 2025, Olofsson 2014 |

A frase sobre Jasansky 2025 no texto descreve só o título e o registro do dataset. Antes de afirmar a lacuna, é preciso ler o
artigo e repetir a busca em bases indexadas.
