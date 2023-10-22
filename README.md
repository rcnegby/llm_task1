### llm task1
to build docker run `docker build -t llm .`

to run docker run `docker run -p 8080:8080 llm`

Модель и данные нужно положить в папки models и data соответсвенно 

Для отправки запроса переходим на http://127.0.0.1:8080/docs и нажимаем POST -> Try it out

на CPU 11th Gen Intel© Core™ i7-11700K @ 3.60GHz × 8 один запрос выполняется примерно 10 сек.
