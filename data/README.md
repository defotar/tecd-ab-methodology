# Данные

[T-ECD](https://huggingface.co/datasets/t-tech/T-ECD) — датасет Т-Банка, лицензия CC BY-NC-SA 4.0. В репозиторий не включён.

Нужна версия `dataset/small`: события доменов `marketplace` и `retail` (дни 1082–1308, 454 файла, ~3 ГБ), а также `users.pq` и `brands.pq`. Список файлов с размерами — `files.txt`.

```bash
sh data/download.sh
```

Скрипт качает в 8 потоков, пропускает уже скачанные файлы, докачивает оборванные и в конце пишет, сколько файлов не скачано. Если не ноль — запустить ещё раз.

Структура событий: `timestamp` (микросекунды от начала отсчёта датасета, день = `timestamp // 86 400 000 000`), `user_id`, `item_id`, `subdomain`, `action_type`, `os`. В ритейле `action_type` — `view`, `added_to_cart`, `click`, `order` (одна строка на купленный товар), в маркетплейсе — `view`, `click`, `clickout`, `like`.
