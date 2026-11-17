# CORTEX — laboratório de design

Protótipo visual independente. Abra `index.html` no navegador. Não exige instalação, internet, build ou integração com o projeto CORTEX.

## Nome do produto

O produto se chama **CORTEX** e continuará com esse nome. ORBIT é somente a origem de ideias discutidas anteriormente, que serão incorporadas ao CORTEX. Não é uma renomeação, um produto substituto ou uma migração para outro assistente.

## Lei de design aprovada pelo usuário

A direção oficial é **Aranha escuro**. Preservar todos os elementos da proposta visual, incluindo os **símbolos e rótulos de Conversar, Criar e Continuar**. Não remover ou substituir elementos sem solicitação explícita do usuário.

Preservar: marca; nova conversa; Início, Projetos e Conversas; projetos CORTEX e Pessoal; conversas recentes; indicação local/privado; núcleo; estado do núcleo; saudação; os três modos e seus símbolos; cartão de retomada; conversa com avatar; campo de mensagem; anexos; microfone; enviar; indicação de modelo local.

## Escopo e isolamento

- Nenhuma integração, chamada de rede, API, modelo, banco de dados ou acesso a arquivos do CORTEX.
- Conteúdo e conversas ilustrativos. Mensagens e preferências existem somente em memória nesta aba e somem ao recarregar.
- Anexos: somente nomes de arquivos selecionados pelo usuário. Não há leitura de conteúdo nem envio.
- Voz: animação simulada; não solicita nem acessa microfone.
- Não existe autenticação ou garantia técnica de privacidade de um produto final; o selo é parte do layout de demonstração.

## Núcleo

Canvas 2D procedural, sem imagem pronta, bibliotecas ou dependências. Esfera com 165 pontos e conexões próximas, órbitas, anéis segmentados, centro vermelho e partículas em trânsito. O desenho é uma aproximação animada da referência aprovada, não uma reprodução pixel a pixel da imagem gerada.

Início: 230–350 px conforme a janela. Chat: 154–190 px. Transição de tamanho mantém o mesmo componente e geometria. Estados: repouso, pensando, ouvindo e respondendo; todos são simulações. Limite aproximado de 30 quadros por segundo, densidade máxima de pixels 2, pausa com aba oculta, respeito à preferência do sistema para reduzir movimento. Painel de ajustes oferece controle manual de movimento e intensidade. Tema claro é uma opção exploratória adicional; Aranha escuro é o padrão.

## Como experimentar

1. Abra `index.html`.
2. Clique em Retomar projeto ou envie uma mensagem para visualizar o núcleo compacto.
3. Durante a resposta simulada, o botão enviar vira parar.
4. Use o botão da Central do CORTEX no canto superior direito. Escolha uma métrica para abrir seu painel; Retomar central volta à lista. Aparência e núcleo mantém os controles do protótipo disponíveis como opção secundária.
5. Use o ícone de sol para comparar o tema claro.
6. Cmd/Ctrl + K abre a conversa e foca o campo; Enter envia; Shift + Enter quebra linha.

## Arquivos

- `index.html`: estrutura, conteúdo e símbolos SVG.
- `style.css`: paleta, composição, tamanhos e responsividade.
- `app.js`: animação procedural e interações demonstrativas.
- `DESIGN.md`: regras e instruções deste laboratório.

## Central do CORTEX

O botão superior direito abre uma janela sobre a conversa. Opções: Internet lease, Net, Services, Studier, Audit log, Costs, Voz e Memória. Selecionar uma opção fecha a central e abre apenas o painel correspondente. O botão **Retomar central** volta à lista e devolve o foco à opção escolhida. Fechar ou pressionar Escape retorna à interface, sem remontar a conversa.

Todos os valores são amostras estáticas da captura de referência: closed, offline, 4/5, 0 fila/0 review, 8 hoje, 11 mês, 0 hoje e 586 documentos. Nenhuma moeda é inferida para Costs. Não há monitoramento ou alteração de permissões reais. Os ajustes do núcleo foram preservados na opção secundária Aparência e núcleo.
