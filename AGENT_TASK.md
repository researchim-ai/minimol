# Задание агенту: Minimol

Основной исполняемый план: [docs/AGENT_EXECUTION_PLAN.md](docs/AGENT_EXECUTION_PLAN.md).

Выполняй его по зависимостям, начиная с E0. Сохрани специализированную линейку
Reg/ADMET/CDS/Rank и добавь отдельную Jev-подобную ветку Minimol-Decision.
Сначала E1/E2 и исправления R0; длинные обучения разрешены планом только после
соответствующих data/code gates и фиксации конечного compute budget.

Не считать synthetic benchmark подтверждением biological quality; не выбирать
модель по test; не перезаписывать старые runs и текущую работу. Записывать
результаты и task status в `artifacts/execution/`. Публикация данных/весов
не входит в автоматическое выполнение. При NO-GO сохранять отрицательные
результаты и продолжать независимые задачи, не обещать superiority.

Контекст реализации: [Decision_LoRA_and_Benchmark](docs/Decision_LoRA_and_Benchmark.md)
и [Decision_Model_Next_Stage](docs/Decision_Model_Next_Stage.md).

Этот файл — точка входа в задание, не новые постоянные правила для репозитория.
