# PPO

## Постановка задачи
С помощью алгоритма Proximal Policy Optimization нужно обучить политику, решающую две подзадачи

1. Подъем маятника из нижнего положения в верхнее с последующей стабилизацией 
2. Постановка конца маятника в соответствии с заданным положением в мировых координатах(таргета)

## Основные используемые идеи
С имплементацией PPO проблем нет, основная задача состоит в разработке функции награды. Она должна влиять на:  

- стабильное положение в верхних точках и размещение около таргета  
- раскручивание в нижний точках(приложение силы и набирание скорости)  
- стабилизация при раскручивании в верхней точке

Для этой и других моделей установим максимальное время жизни и добавим допустимые границы действия агента [-2, 2]. Агенту не нужно много места и времени для раскачивания и стабилизации, и нет смысла исследовать мир вдоль оси x.

## Функция награды
Начнем с первого пункта. В похожих окружениях агент получает награду, если удерживает объект в пределах 0.2 радиан(угол с горизонталью). Используем самое простое - косинус. Для размещения около таргета используем штраф - норма расстояния от центра текущей системы координат(таргета или начало координат).
```python
reward = np.cos(theta) - np.clip((x - target)**2, 0.0, 1.0)
```
Обучим модель уравновешивать маятник и устанавливать конец в начало системы координат таргета
[![Watch the video](https://github.com/Pikudan/PPO/blob/4c8ac915d27fe7b9a70a311848ba6a1ba831d021/video/%D1%83%D1%80%D0%B0%D0%B2%D0%BD%D0%BE%D0%B2%D0%B5%D1%88%D0%B8%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5.jpg)](https://github.com/Pikudan/PPO/blob/903966790292efba3ba0ebc042a8e169e6d716ec/video/%D1%83%D1%80%D0%B0%D0%B2%D0%BD%D0%BE%D0%B2%D0%B5%D1%88%D0%B8%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5.mov)



Для чего начали с этой модели? Для того чтобы определить подходящее поведение награды и диапозон штрафа, награды(может происходить ситуация, когда падение конца маятника больше уменьшает расстояние чем малое передвижение тележки).

Можно также заметить что агент не стремиться попасть прямо в таргет, а останавливается около него. Можно добавить в окрестности таргета побольше награды, так и поступим далее.

Второй пункт немного сложнее. Мы должны поощрять скорость по направлению вверх, но тогда есть субоптимальная стратегия -  это бесконечно крутиться. C последним будем бороться тем, что при прохождении нижнего части круга суммарная награда дает отрицательную величину. Проще всего будет наказывать за движение вниз, а на движение вверх не реагировать. Также чтобы агент начал пробовать двигаться добавим награду за работу мотора.
```python
reward = min(-theta**2 + 0.1 * dtheta**2 + 0.001 * min(a**2, 1.0) - 10.0, 0.0) + np.exp(-(x - target)**2)
```
В третьем пункте будем наказывать за падение в верхней части и поощрять за возвышение
```python
reward = np.cos(theta) * (1 - dtheta**2) - max(dtheta * theta, 0.0)
```

Итоговый вариант выглядит так:


## Установка

Настройка виртуального окружения
```bash
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

## Программа
```
Контент репозитория
```bash
- main
- video # записи игр
- weight # записи игр
```
Контент файлов (ветка main)
```bash
- main.py
- train.py # train
- test.py # inference 
- envs.py # environment
- agent.py # PPO agent
- network.py # Policy and Value Network
- runners.py # run game in environment
- utils
  - AsArray # transform to ndarray
  - Policy # Policy PPO
  - GAE # Generalized Advantage Estimator
  - TrajectorySampler # samples minibatch
  - NormalizeAdvantages # normalize advantages
  - make_ppo_runner # create runner
  - evaluate # play games on inference
```
