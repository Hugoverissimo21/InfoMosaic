news_sources = [
    ("https://www.rtp.pt/noticias/", "RTP"),
    ("https://www.rtp.pt/", "RTP"),
    ("https://www.rtp.pt/noticias/pais/", "RTP"),
    ("https://www.rtp.pt/noticias/mundo/", "RTP"),
    ("https://www.rtp.pt/noticias/politica/", "RTP"),
    ("https://www.rtp.pt/noticias/economia/", "RTP"),
    ("https://www.rtp.pt/noticias/cultura/", "RTP"),
    ("https://www.rtp.pt/noticias/desporto/", "RTP"),
    ("https://www.rtp.pt/noticias/futebol-nacional/", "RTP"),
    ("https://www.publico.pt/", "Público"),
    ("https://www.publico.pt/sociedade/", "Público"),
    ("https://www.publico.pt/2024/", "Público"),
    ("https://www.publico.pt/2004/", "Público"),
    ("https://www.publico.pt/2023/", "Público"),
    ("https://www.publico.pt/2022/", "Público"),
    ("https://www.publico.pt/2021/", "Público"),
    ("https://www.publico.pt/202", "Público"),
    ("https://www.publico.pt/201", "Público"),
    ("https://www.publico.pt/20", "Público"),
    ("https://www.publico.pt/200", "Público"),
    ("https://www.cmjornal.pt/", "Correio da Manhã"),
    ("https://www.cmjornal.pt/politica/", "Correio da Manhã"),
    ("https://www.cmjornal.pt/exclusivos/", "Correio da Manhã"),
    ("https://www.cmjornal.pt/portugal/", "Correio da Manhã"),
    ("https://www.cmjornal.pt/economia/", "Correio da Manhã"),
    ("https://www.cmjornal.pt/cultura/", "Correio da Manhã"),
    ("https://www.cmjornal.pt/desporto/", "Correio da Manhã"),
    ("https://www.cmjornal.pt/mundo/", "Correio da Manhã"),
    ("https://www.cmjornal.pt/sociedade/", "Correio da Manhã"),
    ("https://www.sapo.pt/", "SAPO"),
    ("https://24.sapo.pt/atualidade/", "SAPO"),
    ("https://24.sapo.pt/economia/", "SAPO"),
    ("https://www.sapo.pt/noticias/desporto/", "SAPO"),
    ("https://www.sapo.pt/noticias/atualidade/", "SAPO"),
    ("https://www.sapo.pt/noticias/", "SAPO"),
    ("https://www.sapo.pt/noticias/economia/", "SAPO"),
    ("https://www.sapo.pt/noticias/entretenimento/", "SAPO"),
    ("https://www.sapo.pt/noticias/viagens/", "SAPO"),
    ("https://www.sapo.pt/noticias/lifestyle/", "SAPO"),
    ("https://www.sapo.pt/noticias/fama/", "SAPO"),
    ("https://www.sapo.pt/noticias/tecnologia/", "SAPO"),
    ("https://www.sapo.pt/noticias/planeta/", "SAPO"),
    ("https://omirante.pt/", "O Mirante"),
    ("https://jornaleconomico.sapo.pt", "SAPO"),
    ("https://jornaleconomico.sapo.pt/noticias/", "SAPO"),
    ("https://www.aeiou.pt/", "AEIOU"),
    ("https://zap.aeiou.pt/", "AEIOU"),
    ("https://zap.aeiou.pt/noticias/", "AEIOU"),
    ("https://ionline.sapo.pt/", "SAPO"),
    ("https://ionline.sapo.pt/2024/", "SAPO"),
    ("https://ionline.sapo.pt/202", "SAPO"),
    ("https://ionline.sapo.pt/20", "SAPO"),
    ("https://ionline.sapo.pt/201", "SAPO"),
    ("https://www.dn.pt/", "Diário de Notícias"),
    ("https://www.dn.pt/d/dinheiro/", "Diário de Notícias"),
    ("https://away.iol.pt/", "IOL"),
    ("https://www.iol.pt/", "IOL"),
    ("https://www.iol.pt/noticias/", "IOL"),
    ("https://www.dinheirovivo.pt/", "Dinheiro Vivo"),
    ("https://jornaleconomico.sapo.pt/", "SAPO"),
    ("https://jornaleconomico.sapo.pt/noticias/", "SAPO"),
    ("https://observador.pt/", "Observador"),
    ("https://observador.pt/especiais/", "Observador"),
    ("https://observador.pt/seccao/politica/", "Observador"),
    ("https://observador.pt/2024/", "Observador"),
    ("https://observador.pt/202", "Observador"),
    ("https://observador.pt/20", "Observador"),
    ("https://observador.pt/201", "Observador"),
    ("https://www.record.pt/", "Record"),
    ("https://www.record.pt/futebol/", "Record"),
    ("https://www.record.pt/internacional/", "Record"),
    ("https://www.tsf.pt/", "TSF"),
    ("https://www.lusa.pt/", "Lusa"),
    ("https://www.lusa.pt/article/", "Lusa"),
    ("https://www.lusa.pt/national/", "Lusa"),
    ("https://www.lusa.pt/economia/", "Lusa"),
    ("https://www.lusa.pt/culture/", "Lusa"),
    ("https://www.lusa.pt/desporto/", "Lusa"),
    ("https://www.lusa.pt/international/", "Lusa"),
    ("https://www.lusa.pt/lusofonia/", "Lusa"),
    ("https://www.noticiasaominuto.com/", "Notícias ao Minuto"),
    ("https://www.noticiasaominuto.com/economia/", "Notícias ao Minuto"),
    ("https://www.noticiasaominuto.com/politica/", "Notícias ao Minuto"),
    ("https://www.noticiasaominuto.com/desporto/", "Notícias ao Minuto"),
    ("https://www.noticiasaominuto.com/fama/", "Notícias ao Minuto"),
    ("https://www.noticiasaominuto.com/pais/", "Notícias ao Minuto"),
    ("https://www.noticiasaominuto.com/mundo/", "Notícias ao Minuto"),
    ("https://www.noticiasaominuto.com/tech/", "Notícias ao Minuto"),
    ("https://www.noticiasaominuto.com/cultural/", "Notícias ao Minuto"),
    ("https://www.noticiasaominuto.com/lifestyle/", "Notícias ao Minuto"),
    ("https://tvi.iol.pt/noticias/", "IOL"),
    ("https://sicnoticias.pt/", "SIC Notícias"),
    ("https://sicnoticias.pt/especiais/", "SIC Notícias"),
    ("https://sicnoticias.pt/desporto/", "SIC Notícias"),
    ("https://sicnoticias.pt/economia/", "SIC Notícias"),
    ("https://sicnoticias.pt/saude-e-bem-estar/", "SIC Notícias"),
    ("https://sicnoticias.pt/pais/", "SIC Notícias"),
    ("https://sicnoticias.pt/mundo/", "SIC Notícias"),
    ("https://www.jornaldenegocios.pt/", "Jornal de Negócios"),
    ("https://www.jornaldenegocios.pt/empresas/", "Jornal de Negócios"),
    ("https://www.jornaldenegocios.pt/economia/", "Jornal de Negócios"),
    ("https://www.jornaldenegocios.pt/economia/europa/", "Jornal de Negócios"),
    ("https://www.jornaldenegocios.pt/opiniao/", "Jornal de Negócios"),
    ("https://www.jn.pt/", "Jornal de Notícias"),
    ("https://sol.sapo.pt/", "SAPO"),
    ("https://sol.sapo.pt/2024/", "SAPO"),
    ("https://sol.sapo.pt/2023/", "SAPO"),
    ("https://sol.sapo.pt/20", "SAPO"),
    ("https://sol.sapo.pt/201", "SAPO"),
    ("https://sol.sapo.pt/202", "SAPO"),
    ("https://expresso.pt/", "Expresso"),
    ("https://expresso.pt/economia/", "Expresso"),
    ("https://tribuna.expresso.pt/", "Expresso"),
    ("https://expresso.pt/opiniao/", "Expresso"),
    ("https://expresso.pt/inimigo-publico/", "Expresso"),
    ("https://expresso.pt/politica/", "Expresso"),
    ("https://expresso.pt/sociedade/", "Expresso"),
    ("https://expresso.pt/internacional/", "Expresso"),
    ("https://expresso.pt/cultural/", "Expresso"),
    ("https://expresso.pt/sustentabilidade/", "Expresso"),
    ("https://expresso.pt/boa-cama-boa-mesa/", "Expresso"),
    ("https://cnnportugal.iol.pt/", "CNN Portugal"),
    ("https://cnnportugal.iol.pt/guerra/", "CNN Portugal"),
    ("https://cnnportugal.iol.pt/telecomunicacoes/", "CNN Portugal"),
    ("https://www.nit.pt/", "NiT"),
    ("https://www.nit.pt/cultura/", "NiT"),
    ("https://www.nit.pt/comida/", "NiT"),
    ("https://www.nit.pt/compras/", "NiT"),
    ("https://www.nit.pt/fit/", "NiT"),
    ("https://www.nit.pt/comer-fora/", "NiT"),
    ("https://www.nit.pt/fora-de-casa/", "NiT"),
    ("https://www.nit.pt/listagem/nitvinhos/", "NiT"),
    ("https://www.nit.pt/nittravel/", "NiT"),
    ("https://www.nit.pt/nitcom/", "NiT"),
    ("https://www.nit.pt/escapadinhas-nit/", "NiT"),
    ]