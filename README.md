# Proximal Policy Optimization

## Постановка задачи
С помощью алгоритма Proximal Policy Optimization нужно обучить политику, решающую следующие подзадачи

1. Подъем маятника из нижнего положения в верхнее с последующей стабилизацией 
2. Постановка конца маятника в соответствии с заданным положением в мировых координатах - далее **таргет**
3. Стабилизация маятника с произвольной массой

С имплементацией PPO проблем нет. Основная задача состоит в разработке функции награды и определении данных, доступных моделям. 

## Теория
### Proximal Policy Optimization

Этот алгоритм, основанный на статье [Proximal Policy Optimization Algorithms](https://arxiv.org/pdf/1707.06347),  был реализован на основе кода к домашней работе из  [курса ШАДА](https://github.com/yandexdataschool/Practical_RL/tree/master/week09_policy_II). Также написание алгоритма опиралось на [следующие лекции](https://github.com/FortsAndMills/RL-Theory-book/blob/main/RL_Theory_Book.pdf)

### Наблюдения

Следующий момент состоит в том, как можно определить состояние.
  
Базовы  вектор наблюдений:

- положение тележки относительно таргета - $х - target$ м
- угол маятника с вертикалью - $\theta$ радиан
- угловая скорость тележки - $\frac{dx}{dt}$ м/с
- скорость конца маятника - $\frac{d\theta}{dt}$ рад/с

Расширенный вектор наблюдений:

- положение тележки - $x$ м
- угол маятника с вертикалью - $\theta$ радиан
- скорость тележки - $\frac{dx}{dt}$ м/с
- угловая скорость конца маятника - $\frac{d\theta}{dt}$ рад/с
- положение таргета - $target$ м
  
Для решения задачи с произвольной массой будем давать модели несколько последовательных наблюдений с последнего действия. Так можно дать возможность по данным движений предсказать соотношение масс тележки и маятника.

### Функция награды

Функция награды в этой задач должна влиять на:  

- размещение конца маятника около таргета  
- раскручивание в нижний точках(приложение силы и достижение нужной скорости для подъема)  
- стабилизацию при раскручивании в верхней точке

Начнем с первого пункта. В похожих окружениях агент получает награду, если удерживает объект в пределах 0.2 радиан(угол с вертикалью). Используем самое простое - косинус. Для размещения около таргета используем поощрение/штраф -  функцию от нормы расстояния от центра текущей системы координат(таргета или начала координат).

```python
relative_posistion_cart = x - target + np.sin(theta) * len_pole
```

```python
reward = np.cos(theta) + np.exp(-np.abs(relative_posistion_cart))
```

```python
reward = np.cos(theta) + np.exp(-np.abs(relative_posistion_cart)) * (relative_posistion_cart) < 0.1)
```

```python
reward = np.cos(theta) + 1 - np.abs(relative_posistion_cart)
```

Для чего начали с этой модели? Для того чтобы:

-  определить подходящее поведение награды в пределах 0.2 радиан
-  выбрать L1 или L2 норму
-  выбратьпоощрение в небольшей окрестности или везде
-  сравнить поведение модели  на двух вариантах определения состояния
-  настроить гиперпараметры (количество эпох, $\lambda$)



Второй пункт немного сложнее. Мы должны поощрять скорость по направлению вверх, но тогда есть субоптимальная стратегия -  это бесконечно крутиться. C последним будем бороться тем, что при прохождении нижнего части круга суммарная награда дает неположительную величину. Проще всего будет наказывать за движение вниз, а на движение вверх не реагировать. Также чтобы агент начал пробовать двигаться добавим награду за работу мотора.
```python
reward = min(-theta**2 + 0.1 * dtheta**2 + 0.001 * a**2, 0.0)
```
В третьем пункте будем наказывать за падение в верхней части и поощрять за возвышение
```python
reward = np.cos(theta) - max(dtheta * theta, 0.0)
```

Итоговый вариант функции награды после проведения экспериментов выглядит так:

def get_reward(x, theta, dx, dtheta, target, a, extented_observation, len_pole=0.6):
    if extented_observation:
        pos_cart = np.sin(theta) * len_pole + x - target
    else:
        pos_cart = np.sin(theta) * len_pole + x
        
    if abs(ob[1]) < 0.2:
        # поощрение в ограниченной области
        reward = np.cos(theta) + np.exp(-np.abs(relative_posistion_cart)) * (np.abs(relative_posistion_cart) < 0.1)
        
        # поощрение везде
        # reward = np.cos(theta) + np.exp(-np.abs(relative_posistion_cart))
        
        # штраф за отдаление
        # reward = np.cos(theta) + 1 - np.abs(relative_posistion_cart)
    elif abs(ob[1]) < np.pi / 2:
        # больше нуля при подьеме, меньше нуля при опускании
        reward = np.cos(theta) - max(dtheta * theta, 0.0)
    else:
        reward = np.clip(-theta**2 + 0.1 * dtheta**2 + 0.001 * a**2 , -np.pi**2, 0.0)
    return reward

### Дополнительные параметры


Для этой и других моделей установим максимальное время жизни. Для агентов, чья задача ограничивается размещением около таргета, добавим допустимые углы маятника с вертикалью в 0.2 радиана. Агенту не нужно много времени для раскачивания и стабилизации, и нет смысла исследовать мир вдоль оси x.

Для задачи с уравновешиванием маятника с произвольной массой. Масса маятника, по умолчанию установленная в среде, менялась максимум в 10 раз(уменьшение и увеличение).

## Эксперименты

### Постановка конца маятника в соответствии с заданным положением в мировых координатах - далее **таргет**

В начале обучались агенты для уравновешивания маятника и постановки конца маятника в таргет. Вариант с расширенным вектором наблюдений давал стабильнее обучение чем с базовым вектором наблюдений но сходился дольше. Количество эпох из этих экспериментов нужно было брать небольшое - 8-32. 

На видео можно сравнить поведение моделей с
[L1](https://github.com/Pikudan/PPO/blob/51f794d113a53735c0851f2a9985dbfdd922debc/video/%D1%83%D1%80%D0%B0%D0%B2%D0%BD%D0%BE%D0%B2%D0%B5%D1%88%D0%B8%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5_L1.mov)
и
[L2](https://github.com/Pikudan/PPO/blob/51f794d113a53735c0851f2a9985dbfdd922debc/video/%D1%83%D1%80%D0%B0%D0%B2%D0%BD%D0%BE%D0%B2%D0%B5%D1%88%D0%B8%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5_L2.mov)
 нормой. Можно также заметить что агент не стремиться попасть прямо в таргет, а останавливается около него. $L_1$ помогает лучше помогает достигать лучших результатов.


Также важным оказалось
[ограничение](https://github.com/Pikudan/PPO/blob/51f794d113a53735c0851f2a9985dbfdd922debc/video/%D1%83%D1%80%D0%B0%D0%B2%D0%BD%D0%BE%D0%B2%D0%B5%D1%88%D0%B8%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5_L1.mov)
области получения награды за близость к таргету - агент быстрее хотел попасть туда.
[Без ограничения](https://github.com/Pikudan/PPO/blob/51f794d113a53735c0851f2a9985dbfdd922debc/video/%D1%83%D1%80%D0%B0%D0%B2%D0%BD%D0%BE%D0%B2%D0%B5%D1%88%D0%B8%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5_L2.mov)
агент возможно хотел снизить вероятность падения и действовал осторожно. Но как было замечено далее для задачи upswing ограничение только мешало моделе понять куда надо переместить маятник.

### Подъем маятника из нижнего положения в верхнее с последующей стабилизацией 
Далее обучение стартовало уже с нижней точки c вышепоказаной моделью. Расширенный вектор наблюдений здесь уже лучше справлялся. По датасету нужно пройтись при этом несколько раз, теоретически — хочется как можно больше, но понятно, что чем больше расходится $\pi_{\theta}$ и $\pi_{old}$, тем менее эффективна «нижняя оценка» и тем больше данных будет резать наш клиппинг. Соответственно, количество эпох — сколько раз пройтись по датасету — является ключевым гиперпараметром. Занятно было видеть этого
[агента](https://github.com/Pikudan/PPO/blob/95f32af3cb241d87a755bb91b10c7e804ff4bebd/video/8epochs.mov)
, обученного с 8 эпохами. Он поднимал маятник, но не следовал к таргету. Остальные сильно себя раскручивали и несколько раз крутились, но все же выполняли обе первые подзадачи.

Cледующий
[агент](https://github.com/Pikudan/PPO/blob/95f32af3cb241d87a755bb91b10c7e804ff4bebd/video/16epochs.mov)
получился одним из самых хороших. Дальнейшие шаги улучшения моделей сводяться только к подбору параметров.

### Стабилизация маятника с произвольной массой

Рассматривались варианты с 4, 5, 10 последовательными наблюдениями, получаемые агентом. На 10 агенту было сложно учится. Оптимальнее было брать 5 последовательных наблюдений.

За таргетом модели не следовали. Это было решено изменением награды - наказывать за отдаление. Похоже из-за неопределенности массы, они предпочитали безопасно находиться наверху, а не искать награду. Также измененялось количество наблюдений. 

Вот примеры работы агента со штрафом за отдаление от таргета с [0.5](https://github.com/Pikudan/PPO/blob/8f303d2fe9cbf6306ed4f69469ad9840832769d5/video/05mass.mov), [1.0](https://github.com/Pikudan/PPO/blob/8f303d2fe9cbf6306ed4f69469ad9840832769d5/video/1mass.mov), [5.0](https://github.com/Pikudan/PPO/blob/8f303d2fe9cbf6306ed4f69469ad9840832769d5/video/5mass.mov), [50.0](https://github.com/Pikudan/PPO/blob/8f303d2fe9cbf6306ed4f69469ad9840832769d5/video/50mass.mov)  единиц масс. Хоть агент и уравновешивает различные массы, видно что он не может предсказать ее по его хаотичным движениям. Были попытки обучить отдельную модель предсказывать вес маятника по первым наблюдениям, но видно физика этой среды сложнее - модель предсказывала среднюю массу.

Задачу upswing и уравновешивания вместе решить не удалось для произвольной массы. Отдельный диапозон масс агент все же [поднимал и удерживал](https://github.com/Pikudan/PPO/blob/9924e9647680c456a2df401e92c5a896fb31ea0c/video/upswing_with_mass.mov). Но остальные мог только раскрутить.



## Репозиторий

Контент репозитория
```bash
- main
- video # записи игр
- weight # веса моделей
```
Контент файлов (ветка main)
```bash
- main.py
- train.py   # train
- test.py    # inference 
- envs.py    # environment
- agent.py   # PPO agent
- network.py # Policy and Value Network
- runners.py # run game in environment
- utils
  - AsArray               # Transform to ndarray
  - Policy                # Policy PPO
  - GAE                   # Generalized Advantage Estimator
  - TrajectorySampler     # Samples minibatch
  - NormalizeAdvantages   # Normalize advantages
  - make_ppo_runner       # Create runner
  - evaluate              # Play games on inference
- arguments  # Arguments parse
- writer     # Create a SummaryWriter object for logging
```

## Использование

Настройка виртуального окружения
```bash
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

Для тренировки:
```bash
mjpython main.py --mode=train --upswing=True --extended_observation=True
```

Для тестирования:
```bash
mjpython main.py --mode=test --upswing=True --extended_observation=True --policy_model=policy.pth --value_model=value.ptр
```

```bash
Options:
    --mode                    str       train, test - тренировка, тестирование
    --policy_model            str       путь до весов актера модели
    --value_model             str       путь до весов критика модели
    --upswing                 bool      флаг включения подзадачи upswing (default setting = False)
    --target                  bool      флаг включения подзадачи установки конца маятника в заданных координатах (default setting = False)
    --extended_observation    bool      флаг расширения размерности наблюдений до 5 (default setting = False)
    --mass_use                bool      флаг использования модели для маятника с произвольной массой (default setting = False)
    --mass                    float     масса маятника. Используется только при mode=test. По умолчанию береться из созданной среды
    --gamma                   float     gamma
    --lambda                  float     lambda
    --num_minibatches         int       размер батча траекторий
    --num_observations        int       количество последовательных наблюдений, произошедших между действиями и переданных агенту. С осторожностью устанавливайте - могут быть проблемы с точностью определения временного шага между наблюдениями.
    --num_epochs              int       количество проходов по траектории
    --num_runner_steps        int       количество шагов в среде

```
