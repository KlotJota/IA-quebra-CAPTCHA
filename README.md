# IA-quebra-CAPTCHA
<b>O arquivo final para a execução do projeto é o "usando_captcha.py"</b>

Nesse projeto em Python, foi utilizado a biblioteca de rede neural Keras com o intuito de fazer com que a IA aprenda a ler prints de CAPTCHA em texto e decifre as letras.

Para a construção desse projeto, de início é necessário realizar o tratamento das imagens de CAPTCHA, eliminando o máximo de ruídos que a imagem possui e 
tornando as letras o mais visíveis possível. Para isso, foram utilizados bibliotecas para o tratamento dessas imagens e de arquivos, tais como OpenCV, imutils, pickle, 
glob, entre outras.

De modo geral, a rede neural utilizada no projeto aprende através de uma base de dados de diversas outras imagens de CAPTCHAS, portanto, quanto maior a quantia de arquivos
contidos nessa base, mais acertos a IA irá obter no fim.

Abaixo, deixarei instruçõe sobre a criação de pastas relacionadas ao projeto, desde a criação da base de dados até sua execução. Caso você queira pular essa parte e
utilizar o modelo de IA treinado que deixei pronto nos arquivos, crie apenas uma pasta chamada "captchas_resolver" e coloque nela a imagem (ou imagens) de CAPTCHA que você deseja resolver.

# Criação da base de dados da IA

O arquivo "modelo_CAPTCHA_treinado.hdf5" possui o estado da rede neural treinada com uma base de dados relativamente pequena, portanto a quantidade de acertos não chega 
a 100%. Caso desejar, adicione prints de CAPTCHA à uma pasta de base de dados (sugiro o uso do nome originalmente usado: bdcaptcha), crie também uma pasta onde as imagens
finalizadas serão armazenadas (nome original: finalizadas). Após isso, execute o arquivo "trata_captchas.py". Com isso, você terá todas as imagens (anteriormente obstruídas)
visivelmente mais "limpas" na pasta "finalizadas".

Após a execução do passo acima, você deverá partir para a execução do código "separa_letras.py". Antes disso, crie as pastas "letras" e "identificado", uma delas 
armazenará as letras das imagens separadas e a outra armazenará as imagens por completo, delimitando um retangulo verde ao redor de cada letra, demonstrando o contorno
que o código usou de base para o recorte das letras. Com isso, execute o código "separa_letras.py".

Com as letras separadas e já na pasta "letras", agora deve ser realizado a criação da base de dados. Para isso, foi utilizado um processo manual, então notoriamente exigirá
mais tempo. Crie uma pasta chama "base_letras" e dentro desta pasta crie uma pasta para cada letra do alfabeto (números também, caso você desejar tratar CAPTCHAS que contém
número). Após isso, você recortará todas as letras contidas na pasta "letras" e trará para essa nova pasta, ela servirá como aprendizado para a IA.

# Treino da IA

Com a base dados já criada e todos os cuidados tomados, execute o arquivo "treina_modelo", ele irá treinar a IA utilizando como base de ensino todos os arquivos presentes
na pasta "base_letras" e criará um arquivo originalmente chamado "modelo_CAPTCHA_treinado.hdf5". Ele será o último estado da nossa IA e será utilizado para decifrar os 
CAPTCHAS daqui pra frente.

# Execução do projeto

Por fim, crie uma última pasta chamada "captchas_resolver" e coloque dentro dela toda e qualquer imagem de CAPTCHA que você querer resolver. Após, execute o arquivo 
"usando_captcha.py" e pronto, ele realizará o tratamento completo da imagem e irá decifrar as letras presentes na imagem e retorná-las no terminal.


Este código foi construído seguindo as aulas do canal Hashtag Programação, no YouTube.



