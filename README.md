# readme

data01: uses arquivo CDX api to extract all websites from a given domain and save into /data/url/ and data/urls.csv

---

docker build -t infomosaic .  

docker run --rm infomosaic

docker run --rm -v $(pwd)/data:/app/data infomosaic 

fcd- e se criar uma database com todas as noticias e pre processar o sentimento de cada e dps assim so tnh de procurar as keyowrds e calcular a media do sentimento ou assim e os pesos etc. dps podia ter uma database q tem as notifiaias como key e o value é o set de palavras, dps a dtree avaliava cada noticia a cer se era aprovada e se sim dps é q calculava as coisas todas

---

- some news ids might be repeated. should look iinto that and fix it when finish news extraction, ficar com o id mais antigo

- should have used a database to store all the data

- should have processed the set of words for each new will scarrping, would save time later on (implies changing data02)